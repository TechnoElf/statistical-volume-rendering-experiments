#include "common.slang"

RWTexture2D<float4> render_texture;

uniform float threshold;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint2 dim;
    render_texture.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim)) { return; }

    // Generate ray from camera
    float2 uv = (float2(tid.xy)) / float2(dim);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y for typical screen coordinates

    float4 clip_near = float4(ndc, 0.0, 1.0);
    float4 clip_far = float4(ndc, 1.0, 1.0);

    float4 view_near = mul(inv_projection, clip_near);
    float4 view_far = mul(inv_projection, clip_far);
    view_near /= view_near.w;
    view_far /= view_far.w;

    float4 world_near = mul(inv_view, view_near);
    float4 world_far = mul(inv_view, view_far);

    float3 ray_origin = world_near.xyz;
    float3 ray_dir = normalize(world_far.xyz - world_near.xyz);

    // Transform ray to volume space (model transforms unit cube to world)
    float3 local_origin = mul(inv_model, float4(ray_origin, 1.0)).xyz;
    float3 local_dir = normalize(mul((float3x3)inv_model, ray_dir));

    // Intersect with volume bounding box
    float t_near, t_far;
    float4 color = float4(0);

    if (intersectBox(local_origin, local_dir, t_near, t_far)) {
        t_near = max(t_near, 0.0);

        float sigma_t = sigma_a + sigma_s;
        float max_density = 1.0; // Assume densities are normalized to [0,1]

        float3 throughput = float3(1.0);
        float3 pos = local_origin + local_dir * t_near;
        float3 dir = local_dir;
        float remaining_dist = t_far - t_near;

        float step_size = remaining_dist / 512.0;

        for (int i = 0; i < 512; i++) {
            float t = (float(i) + 0.5) * step_size;
            float3 step_pos = pos + dir * t;
            float density = sampleDensityNoMap(step_pos);

            if (density > threshold) {
                float depth = clamp((511.5 * step_size) - t, 0, 1);
                float3 gradient = float3(
                    density - sampleDensityNoMap(step_pos - float3(1.0 / 512.0, 0, 0)),
                    density - sampleDensityNoMap(step_pos - float3(0, 1.0 / 512.0, 0)),
                    density - sampleDensityNoMap(step_pos - float3(0, 0, 1.0 / 512.0))
                );
                float3 normal = (-normalize(gradient) + float3(1)) / 2.0;
                color = float4(normal, depth);
                break;
            }
        }
    }

    render_texture[tid.xy] = color;
}
