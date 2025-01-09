#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "tira/image.h"
#include "tira/parser.h"
#include <tira/graphics/camera.h>
#include "device_kernels.cuh"  // to include the device header

// Structure definitions
struct sphere {
    float radius;
    glm::vec3 position;
    glm::vec3 color;
};

struct plane {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct light {
    glm::vec3 position;
    glm::vec3 color;
};

int main(const int argc, char* argv[]) {
    std::string in_inputname, in_outputname = "output_file.bmp";

    std::cout << "Enter the scene file name: ";
    std::cin >> in_inputname;

    // Parse scene
    tira::parser Parser(in_inputname);

    
    std::vector<void*> objects;
    std::vector<light> lights;
    
    
    const size_t S = Parser.count("sphere");
    for (size_t si = 0; si < S; si++) {
        struct sphere* s = new sphere();
        s->radius = Parser.get<float>("sphere", si, 0);
        s->position[0] = Parser.get<float>("sphere", si, 1);
        s->position[1] = Parser.get<float>("sphere", si, 2);
        s->position[2] = Parser.get<float>("sphere", si, 3);
        s->color.r = Parser.get<float>("sphere", si, 4);
        s->color.g = Parser.get<float>("sphere", si, 5);
        s->color.b = Parser.get<float>("sphere", si, 6);

        objects.push_back(s);
    }

    
    const size_t P = Parser.count("plane");
    for (size_t pi = 0; pi < P; pi++) {
        struct plane* p = new plane();
        p->position[0] = Parser.get<float>("plane", pi, 0);
        p->position[1] = Parser.get<float>("plane", pi, 1);
        p->position[2] = Parser.get<float>("plane", pi, 2);
        p->normal[0] = Parser.get<float>("plane", pi, 3);
        p->normal[1] = Parser.get<float>("plane", pi, 4);
        p->normal[2] = Parser.get<float>("plane", pi, 5);
        p->color.r = Parser.get<float>("plane", pi, 6);
        p->color.g = Parser.get<float>("plane", pi, 7);
        p->color.b = Parser.get<float>("plane", pi, 8);

        objects.push_back(p);
    }

    
    const size_t L = Parser.count("light");
    for (size_t li = 0; li < L; li++) {
        light l;
        l.position[0] = Parser.get<float>("light", li, 0);
        l.position[1] = Parser.get<float>("light", li, 1);
        l.position[2] = Parser.get<float>("light", li, 2);
        l.color.r = Parser.get<float>("light", li, 3);
        l.color.g = Parser.get<float>("light", li, 4);
        l.color.b = Parser.get<float>("light", li, 5);

        lights.push_back(l);
    }

    
    tira::camera camera;
    glm::vec3 cpos;
    cpos[0] = Parser.get<float>("camera_position", 0);
    cpos[1] = Parser.get<float>("camera_position", 1);
    cpos[2] = Parser.get<float>("camera_position", 2);
    camera.position(cpos);

    glm::vec3 clook;
    clook[0] = Parser.get<float>("camera_look", 0);
    clook[1] = Parser.get<float>("camera_look", 1);
    clook[2] = Parser.get<float>("camera_look", 2);
    camera.lookat(clook);
    camera.fov(Parser.get<float>("camera_fov", 0));

    
    glm::vec3 background;
    background[0] = Parser.get<float>("background", 0);
    background[1] = Parser.get<float>("background", 1);
    background[2] = Parser.get<float>("background", 2);

    
    const auto width = Parser.get<unsigned int>("resolution", 0);
    const auto height = Parser.get<unsigned int>("resolution", 1);

    
    unsigned char* d_image;
    void** d_objects;
    light* d_lights;

    
    cudaMalloc(&d_image, width * height * 3);
    cudaMalloc(&d_objects, objects.size() * sizeof(void*));
    cudaMalloc(&d_lights, lights.size() * sizeof(light));

    
    cudaMemcpy(d_objects, objects.data(), objects.size() * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(light), cudaMemcpyHostToDevice);

    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    render_kernel<<<gridDim, blockDim>>>(
        d_image, width, height, camera, background, 
        d_objects, objects.size(), d_lights, lights.size()
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Rendering time: " << milliseconds << " ms" << std::endl;

   
    std::vector<unsigned char> h_image(width * height * 3);
    cudaMemcpy(h_image.data(), d_image, width * height * 3, cudaMemcpyDeviceToHost);
    
    tira::image<unsigned char> Image(width, height, 3, h_image.data());
    Image.save(in_outputname);


    cudaFree(d_image);
    cudaFree(d_objects);
    cudaFree(d_lights);


    for (auto obj : objects) {
        delete obj;
    }

    return 0;
}
