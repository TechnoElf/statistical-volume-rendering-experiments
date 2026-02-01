Texture3D<float> densities;
SamplerState density_sampler;

uniform uint density_grid_size;
uniform uint frame_index;

uniform float4x4 model;
uniform float4x4 inv_model;
uniform float4x4 view;
uniform float4x4 inv_view;
uniform float4x4 projection;
uniform float4x4 inv_projection;

uniform float3 light_dir;
uniform float3 light_color;

uniform float sigma_a;  // absorption coefficient
uniform float sigma_s;  // scattering coefficient

// PCG random number generator
uint pcg(inout uint state) {
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

float rand(inout uint state) {
    return float(pcg(state)) / 4294967295.0;
}

// Sample density from the 3D texture (assumes volume is in [0,1]^3)
float sampleDensity(float3 pos) {
    if (any(pos < 0.0) || any(pos > 1.0)) {
        return 0.0;
    }
    return densities.SampleLevel(density_sampler, pos, 0);
}

// Transform point from world space to volume space
float3 worldToVolume(float3 world_pos) {
    float3 local = mul(inv_model, float4(world_pos, 1.0)).xyz;
    // Assuming the model transforms the unit cube [0,1]^3
    return local;
}

// Ray-box intersection for unit cube [0,1]^3
bool intersectBox(float3 origin, float3 dir, out float t_near, out float t_far) {
    float3 inv_dir = 1.0 / dir;
    float3 t0 = (0.0 - origin) * inv_dir;
    float3 t1 = (1.0 - origin) * inv_dir;

    float3 t_min = min(t0, t1);
    float3 t_max = max(t0, t1);

    t_near = max(max(t_min.x, t_min.y), t_min.z);
    t_far = min(min(t_max.x, t_max.y), t_max.z);

    return t_near <= t_far && t_far > 0.0;
}

// Henyey-Greenstein phase function
float phaseHG(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * denom * sqrt(denom));
}

// Sample direction from Henyey-Greenstein phase function
float3 sampleHG(float3 wo, float g, inout uint rng) {
    float u1 = rand(rng);
    float u2 = rand(rng);

    float cos_theta;
    if (abs(g) < 1e-3) {
        cos_theta = 1.0 - 2.0 * u1;
    } else {
        float sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    }

    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float phi = 2.0 * 3.14159265 * u2;

    // Create local coordinate frame
    float3 w = -wo;
    float3 u = normalize(cross(abs(w.x) > 0.1 ? float3(0, 1, 0) : float3(1, 0, 0), w));
    float3 v = cross(w, u);

    return normalize(sin_theta * cos(phi) * u + sin_theta * sin(phi) * v + cos_theta * w);
}

// Transmittance along a ray using ray marching
float transmittance(float3 origin, float3 dir, float max_dist, float sigma_t, int num_steps) {
    float step_size = max_dist / float(num_steps);
    float tau = 0.0;

    for (int i = 0; i < num_steps; i++) {
        float t = (float(i) + 0.5) * step_size;
        float3 pos = origin + dir * t;
        float density = sampleDensity(pos);
        tau += density * sigma_t * step_size;
    }

    return exp(-tau);
}

// Delta tracking (Woodcock tracking) for free-flight sampling
bool sampleDistance(float3 origin, float3 dir, float t_max, float sigma_t, float max_density,
                    inout uint rng, out float t, out float3 pos) {
    float sigma_maj = sigma_t * max_density;
    t = 0.0;

    const int MAX_ITERATIONS = 1000;
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        t -= log(1.0 - rand(rng)) / sigma_maj;

        if (t >= t_max) {
            return false; // Escaped the volume
        }

        pos = origin + dir * t;
        float density = sampleDensity(pos);

        if (rand(rng) < density * sigma_t / sigma_maj) {
            return true; // Real collision
        }
    }

    return false;
}

// Estimate direct lighting at a point in the volume
float3 estimateDirectLight(float3 pos, float sigma_t, float max_density) {
    // Transform light direction to volume space for shadow ray
    float3 light_dir_normalized = normalize(light_dir);
    float3 local_light_dir = normalize(mul((float3x3)inv_model, light_dir_normalized));

    // Ray march to estimate transmittance towards light
    float t_near, t_far;
    if (!intersectBox(pos, local_light_dir, t_near, t_far)) {
        return light_color;
    }

    float max_dist = max(0.0, t_far);
    float trans = transmittance(pos, local_light_dir, max_dist, sigma_t, 32);

    return light_color * trans;
}
