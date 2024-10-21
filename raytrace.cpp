#include <iostream>
#include <vector>
#include <chrono>
#include <glm/glm.hpp>
#include "tira/image.h"
#include "tira/parser.h"
#include <tira/graphics/camera.h>

#include <thread> //C++ thread library

std::string in_outputname = "output_file.bmp";
unsigned int in_blocksize = 16;

struct ray {
    glm::vec3 origin;
    glm::vec3 direction;
    unsigned int order = 0;
};

class object;

struct hit {
    object* obj;
    glm::vec3 pos;
    ray r;
    float t;
    glm::vec3 norm;
    glm::vec3 color;
};

struct light {
    glm::vec3 position;
    glm::vec3 color;
};

std::vector<object*> Objects;
tira::camera Camera;

glm::vec3 Background;

class object {
public:
    virtual ~object() = default;

    virtual bool intersect(ray r, hit& h) { return false; }
    virtual bool intersect(ray r, float dist) { return false; }
};

class LightEngine {
    std::vector<light> _lights;

public:
    void addLight(glm::vec3 position, glm::vec3 color) {
        light l;
        l.position = position;
        l.color = color;
        _lights.push_back(l);
    }

    glm::vec3 illuminate(hit h) {
        glm::vec3 sumLight(0.0f, 0.0f, 0.0f);

        for (size_t li = 0; li < _lights.size(); li++) {
            ray lr;
            lr.origin = h.pos;
            glm::vec3 to_light = _lights[li].position - h.pos;
            float light_dist = glm::length(to_light);
            lr.direction = glm::normalize(to_light);

            bool occluded = false;
            for (size_t oi = 0; oi < Objects.size(); oi++) {
                if (Objects[oi] == h.obj) continue;
                hit occlusion;
                occluded = Objects[oi]->intersect(lr, occlusion);
                if (occluded) break;
            }
            if (!occluded) {
                float dot = glm::dot(h.norm, lr.direction);
                if (dot > 0)
                    sumLight += dot * _lights[li].color;
            }
        }
        return h.color * sumLight;
    }
};

LightEngine* Lighting;


class sphere : object {
public:
    float radius;
    glm::vec3 position;
    glm::vec3 color;

    //different
    bool intersect(ray r, hit& h) {
        glm::vec3 oc = r.origin - position;
        float b = glm::dot(oc, r.direction);
        glm::vec3 qc = oc - b * r.direction;
        float disc = radius * radius - glm::dot(qc, qc);
        if (disc < 0.0) return false;
        disc = sqrt(disc);
        float t0 = -b - disc;
        float t1 = -b + disc;

        if (t0 < 0 || t1 < 0) return false;

        h.pos = r.origin + t0 * r.direction;
        h.color = color;
        h.norm = glm::normalize(h.pos - position);
        h.t = t0;
        return true;
    }

    bool intersect(ray r, float dist) {
        glm::vec3 oc = r.origin - position;
        float b = glm::dot(oc, r.direction);
        glm::vec3 qc = oc - b * r.direction;
        float disc = radius * radius - glm::dot(qc, qc);
        if (disc < 0.0) return false;
        disc = sqrt(disc);
        float t0 = -b - disc;
        float t1 = -b + disc;

        if (t0 < 0 || t1 < 0) return false;
        if (t0 < dist) return true;

        return false;
    }

    bool intersect_Pythagorean(ray r, hit& h) {
        glm::vec3 p = position;
        glm::vec3 s = r.origin;
        glm::vec3 v = r.direction;

        float a = glm::dot(v, v);
        float b = 2 * glm::dot(v, s - p);
        glm::vec3 s_p = s - p;
        float c = glm::dot(s_p, s_p) - radius * radius;

        float discriminant = b * b - 4 * a * c;

        float t0 = (-b - sqrt(discriminant)) / (2 * a);
        float t1 = (-b + sqrt(discriminant)) / (2 * a);

        if (t0 <= 0 && t1 <= 0) return false;

        float t;

        if (t0 <= 0) t = t1;
        else if (t1 <= 0) t = t0;
        else t = std::min(t0, t1);

        h.pos = s + t * v;
        h.color = color;
        h.obj = this;
        h.norm = glm::normalize(h.pos - p);
        h.t = t;
        return true;
    }

    bool intersect_Pythagorean(ray r, float dist) {
        glm::vec3 p = position;
        glm::vec3 s = r.origin;
        glm::vec3 v = r.direction;

        float a = glm::dot(v, v);
        float b = 2 * glm::dot(v, s - p);
        glm::vec3 s_p = s - p;
        float c = glm::dot(s_p, s_p) - radius * radius;
        float disc = b * b - 4 * a * c;
        if (disc < 0) return false;
        float t0 = (-b - sqrt(disc)) / (2 * a);
        float t1 = (-b + sqrt(disc)) / (2 * a);

        if (t0 >= 0 && t0 <= dist) return true;
        if (t1 >= 0 && t1 <= dist) return true;

        return false;
    }

};

class plane : public object {
public:
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;

    bool intersect(ray r, hit& h) override {
        float denom = glm::dot(normal, r.direction);  // Check if the ray is parallel to the plane
        if (fabs(denom) > 1e-6) {
            float t = glm::dot(position - r.origin, normal) / denom;
            if (t >= 0) {
                h.pos = r.origin + t * r.direction;
                h.color = color;
                h.obj = this;
                h.norm = normal;
                h.t = t;
                return true;
            }
        }
        return false;  // No intersection
    }

    bool intersect(ray r, float dist) override {
        float denom = glm::dot(normal, r.direction);

        if (fabs(denom) > 1e-6) {
            float t = glm::dot(position - r.origin, normal) / denom;

            if (t >= 0 && t < dist) {
                return true;
            }
        }
        return false;
    }
};


ray pixel2ray(tira::camera c, unsigned int resolution, unsigned int xi, unsigned int yi) {
    float x = (float)xi / (float)resolution - 0.5f;
    float y = -((float)yi / (float)resolution - 0.5f);
    ray result{};

    result.origin = c.position();
    result.direction = c.ray(x, y);
    return result;
}

void loadSpheres(tira::parser parse) {
    const size_t S = parse.count("sphere");

    for (size_t si = 0; si < S; si++) {
        sphere* s = new sphere();
        s->radius = parse.get<float>("sphere", si, 0);
        s->position[0] = parse.get<float>("sphere", si, 1);
        s->position[1] = parse.get<float>("sphere", si, 2);
        s->position[2] = parse.get<float>("sphere", si, 3);
        s->color.r = parse.get<float>("sphere", si, 4);
        s->color.g = parse.get<float>("sphere", si, 5);
        s->color.b = parse.get<float>("sphere", si, 6);

        Objects.push_back((object*)s);
    }
    const size_t P = parse.count("plane");

    for (size_t pi = 0; pi < P; pi++) {
        plane* p = new plane();
        p->position[0] = parse.get<float>("plane", pi, 0);
        p->position[1] = parse.get<float>("plane", pi, 1);
        p->position[2] = parse.get<float>("plane", pi, 2);
        p->normal[0] = parse.get<float>("plane", pi, 3);
        p->normal[1] = parse.get<float>("plane", pi, 4);
        p->normal[2] = parse.get<float>("plane", pi, 5);
        p->color.r = parse.get<float>("plane", pi, 6);
        p->color.g = parse.get<float>("plane", pi, 7);
        p->color.b = parse.get<float>("plane", pi, 8);

        Objects.push_back((object*)p);
    }
}


void loadLights(tira::parser p) {
    Lighting = new LightEngine();
    const size_t L = p.count("light");

    for (size_t li = 0; li < L; li++) {
        glm::vec3 pos;
        glm::vec3 c;

        pos[0] = p.get<float>("light", li, 0);
        pos[1] = p.get<float>("light", li, 1);
        pos[2] = p.get<float>("light", li, 2);
        c.r = p.get<float>("light", li, 3);
        c.g = p.get<float>("light", li, 4);
        c.b = p.get<float>("light", li, 5);

        Lighting->addLight(pos, c);
    }
}

void loadCamera(tira::parser p) {
    glm::vec3 cpos;
    cpos[0] = p.get<float>("camera_position", 0);
    cpos[1] = p.get<float>("camera_position", 1);
    cpos[2] = p.get<float>("camera_position", 2);

    Camera.position(cpos);

    glm::vec3 clook;
    clook[0] = p.get<float>("camera_look", 0);
    clook[1] = p.get<float>("camera_look", 1);
    clook[2] = p.get<float>("camera_look", 2);

    Camera.lookat(clook);

    Camera.fov(p.get<float>("camera_fov", 0));
}

void loadBackground(tira::parser p) {
    Background[0] = p.get<float>("background", 0);
    Background[1] = p.get<float>("background", 1);
    Background[2] = p.get<float>("background", 2);
}

void RenderImagePortion(tira::image<unsigned char>* image, unsigned int start_y, unsigned int end_y) {
    const unsigned int width = image->width();
    const unsigned int height = image->height();
    unsigned int resolution = std::max(width, height);

    for (unsigned int y = start_y; y < end_y; y++) {
        for (unsigned int x = 0; x < width; x++) {
            ray r = pixel2ray(Camera, resolution, x, y);
            unsigned int hits = 0;
            hit closest_hit;
            closest_hit.t = 99999;
            hit h;

            for (size_t oi = 0; oi < Objects.size(); oi++) {
                if (Objects[oi]->intersect(r, h)) {
                    hits++;
                    if (h.t < closest_hit.t) {
                        closest_hit = h;
                    }
                }
            }

            if (hits == 0) {
                (*image)(x, y, 0) = (unsigned char)(Background[0] * 255);
                (*image)(x, y, 1) = (unsigned char)(Background[1] * 255);
                (*image)(x, y, 2) = (unsigned char)(Background[2] * 255);
            }
            else {
                glm::vec3 finalColor = Lighting->illuminate(closest_hit);
                finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
                (*image)(x, y, 0) = (unsigned char)(finalColor[0] * 255);
                (*image)(x, y, 1) = (unsigned char)(finalColor[1] * 255);
                (*image)(x, y, 2) = (unsigned char)(finalColor[2] * 255);
            }
        }
    }
}

void RenderImage(tira::image<unsigned char>* image, int num_threads) {
    const unsigned int height = image->height();
    unsigned int rows_per_thread = height / num_threads;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        unsigned int start_y = i * rows_per_thread;
        unsigned int end_y = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;

        threads.emplace_back(std::thread(RenderImagePortion, image, start_y, end_y));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

int main(const int argc, char* argv[]) {
    
    int num_threads;
	std::string in_inputname;
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;

    std::cout << "Enter the scene file name: "; //spheramid.scene
    std::cin >> in_inputname;


    tira::parser Parser(in_inputname);

    loadSpheres(Parser);
    loadLights(Parser);
    loadCamera(Parser);
    loadBackground(Parser);

    const auto width = Parser.get<unsigned int>("resolution", 0);
    const auto height = Parser.get<unsigned int>("resolution", 1);

    auto start = std::chrono::high_resolution_clock::now();

    tira::image<unsigned char>* Image = new tira::image<unsigned char>(width, height, 3);

    RenderImage(Image, num_threads);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    auto micro_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avgTimePerPixel = static_cast<double>(micro_seconds.count()) / (width * height);

    std::cout << "Render time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Average time per pixel: " << avgTimePerPixel << " microseconds" << std::endl;

    Image->save(in_outputname);
    delete Image;

    return 0;
}