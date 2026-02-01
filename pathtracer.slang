#include "common.slang"

RWTexture2D<float4> render_texture;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint2 dim;
    render_texture.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim)) { return; }

    // Initialize RNG with pixel position and frame index
    uint rng = tid.x + tid.y * dim.x + frame_index * dim.x * dim.y;
    pcg(rng); // Warm up

    // Generate ray from camera
    float2 jitter = float2(rand(rng), rand(rng));
    float2 uv = (float2(tid.xy) + jitter) / float2(dim);
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
    float3 color = float3(0.0, 0.0, 0.0); // Background color

    if (intersectBox(local_origin, local_dir, t_near, t_far)) {
        t_near = max(t_near, 0.0);

        float sigma_t = sigma_a + sigma_s;
        float max_density = 1.0; // Assume densities are normalized to [0,1]
        float g = 0.0; // Isotropic scattering

        float3 throughput = float3(1.0);
        float3 pos = local_origin + local_dir * t_near;
        float3 dir = local_dir;
        float remaining_dist = t_far - t_near;

        float depth_transmittance = transmittance(pos, dir, remaining_dist, sigma_t, 32);
        color = float3(0.0, 1.0, 1.0) * depth_transmittance;

        const int MAX_BOUNCES = 8;
        for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
            float t;
            float3 scatter_pos;

            if (!sampleDistance(pos, dir, remaining_dist, sigma_t, max_density, rng, t, scatter_pos)) {
                // Ray escaped the volume
                break;
            }

            // Update position
            pos = scatter_pos;

            // Russian roulette
            if (bounce > 2) {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if (rand(rng) > p) {
                    break;
                }
                throughput /= p;
            }

            // Compute scattering
            float albedo = sigma_s / sigma_t;
            throughput *= albedo;

            // Add contribution from direct lighting
            float3 direct = estimateDirectLight(pos, sigma_t, max_density);
            float phase = phaseHG(dot(-dir, normalize(light_dir)), g);
            color += throughput * direct * phase;

            // Sample new direction
            dir = sampleHG(dir, g, rng);

            // Update remaining distance
            float new_t_near, new_t_far;
            if (!intersectBox(pos + dir * 0.001, dir, new_t_near, new_t_far)) {
                break;
            }
            remaining_dist = new_t_far;
        }
    } else {
        color = float3(0.0, 1.0, 1.0);
    }

    // Progressive accumulation
    if (frame_index != 0) {
        float4 prev_color = render_texture[tid.xy];
        float weight = 1.0 / float(frame_index + 1);
        float3 accumulated = lerp(prev_color.rgb, color, weight);
        render_texture[tid.xy] = float4(accumulated, 1.0);
    } else {
        render_texture[tid.xy] = float4(color, 1.0);
    }
}
