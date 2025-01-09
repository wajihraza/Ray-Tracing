#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Structure definitions
struct ray {
    glm::vec3 origin;
    glm::vec3 direction;
    unsigned int order = 0;
};

struct hit {
    void* obj; 
    glm::vec3 pos;
    ray r;
    float t;
    glm::vec3 norm;
    glm::vec3 color;
};

struct sphere_data {
    float radius;
    glm::vec3 position;
    glm::vec3 color;
};

struct plane_data {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct light {
    glm::vec3 position;
    glm::vec3 color;
};

// Device function declarations
__device__ bool sphere_intersect(void* obj, ray r, hit& h);
__device__ bool plane_intersect(void* obj, ray r, hit& h);
__device__ glm::vec3 illuminate(hit h, light* lights, int num_lights, void** objects, int num_objects);

// CUDA kernel declaration
__global__ void render_kernel(
    unsigned char* image,
    int width,
    int height,
    glm::vec3 camera_pos,
    glm::vec3 camera_look,
    float fov,
    glm::vec3 background,
    void** d_objects,
    int num_objects,
    light* d_lights,
    int num_lights
);


// Device functions
__device__ bool sphere_intersect(void* obj, ray r, hit& h) {
    sphere_data* sphere = static_cast<sphere_data*>(obj);
    glm::vec3 oc = r.origin - sphere->position;
    float a = glm::dot(r.direction, r.direction);
    float b = 2.0f * glm::dot(oc, r.direction);
    float c = glm::dot(oc, oc) - sphere->radius * sphere->radius;

    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0f * a);
    float t1 = (-b + sqrt_disc) / (2.0f * a);

    float t = (t0 < t1 && t0 > 0) ? t0 : t1;
    if (t < 0) return false;

    h.pos = r.origin + t * r.direction;
    h.color = sphere->color;
    h.norm = glm::normalize(h.pos - sphere->position);
    h.t = t;
    h.obj = obj;
    h.r = r;
    return true;
}

__device__ bool plane_intersect(void* obj, ray r, hit& h) {
    plane_data* plane = static_cast<plane_data*>(obj);
    float denom = glm::dot(plane->normal, r.direction);

    if (fabs(denom) > 1e-6) {
        float t = glm::dot(plane->position - r.origin, plane->normal) / denom;
        if (t >= 0) {
            h.pos = r.origin + t * r.direction;
            h.color = plane->color;
            h.obj = obj;
            h.norm = plane->normal;
            h.t = t;
            h.r = r;
            return true;
        }
    }
    return false;
}

__device__ glm::vec3 illuminate(hit h, light* lights, int num_lights, void** objects, int num_objects) {
    glm::vec3 ambient(0.1f, 0.1f, 0.1f);
    glm::vec3 sumLight = ambient * h.color;

    for (int li = 0; li < num_lights; li++) {
        ray lr;
        lr.origin = h.pos + h.norm * 1e-4f;  
        glm::vec3 to_light = lights[li].position - h.pos;
        lr.direction = glm::normalize(to_light);

        bool occluded = false;
        for (int oi = 0; oi < num_objects; oi++) {
            if (objects[oi] == h.obj) continue;

            hit occlusion;
            bool intersection = sphere_intersect(objects[oi], lr, occlusion);
            if (intersection && occlusion.t < glm::length(to_light)) {
                occluded = true;
                break;
            }
        }

        if (!occluded) {
            float diffuse = glm::max(glm::dot(h.norm, lr.direction), 0.0f);
            sumLight += diffuse * lights[li].color * h.color;
        }
    }

    return glm::clamp(sumLight, 0.0f, 1.0f);
}

// CUDA kernel
__global__ void render_kernel(
    unsigned char* image,
    int width,
    int height,
    glm::vec3 camera_pos,
    glm::vec3 camera_look,
    float fov,
    glm::vec3 background,
    void** d_objects,
    int num_objects,
    light* d_lights,
    int num_lights
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    ray r;
    r.origin = camera_pos;
    r.direction = glm::normalize(camera_look - camera_pos);

    int pixel_index = (y * width + x) * 3;
    glm::vec3 color = background;

    hit closest_hit;
    closest_hit.t = 1e30f;

    for (int oi = 0; oi < num_objects; oi++) {
        hit h;
        if (sphere_intersect(d_objects[oi], r, h)) {
            if (h.t < closest_hit.t) {
                closest_hit = h;
                color = illuminate(h, d_lights, num_lights, d_objects, num_objects);
            }
        }
    }

    image[pixel_index] = (unsigned char)(color.x * 255);
    image[pixel_index + 1] = (unsigned char)(color.y * 255);
    image[pixel_index + 2] = (unsigned char)(color.z * 255);
}


