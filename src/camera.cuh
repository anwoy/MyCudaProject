/*
 * Copyright (C) 2026 Anwoy Kumar Mohanty
 * * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * COMMERCIAL INQUIRIES: For licensing outside of the GPL v3,
 * please contact [anwoy.rkl@gmail.com].
 */

#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <point.cuh>
#include <ray.cuh>
#include <iostream>
#include <vector>
#include <fstream>
#include <utilities.cuh>
#include <chrono>
#include <lodepng.h>

using namespace std;

__global__ void kernel(
    Point* image,
    const int image_width,
    const int image_height,
    const Point pixel00_position,
    const Point delta_w,
    const Point delta_h,
    const Point camera_position,
    Point* star_map,
    const int sm_image_width,
    const int sm_image_height,
    double* disk_map,
    const int disk_map_x,
    const int disk_map_y,
    const int disk_map_z,
    const double time,
    curandState* d_states,
    const int num_samples_per_pixel,
    const bool create_disk,
    const double probty_constant
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = image_width * image_height;
    if (r >= total_pixels)
        return;
    const int j = r / image_width, i = r % image_width;
    const Point pixel = pixel00_position + i*delta_w + j*delta_h;
    Point s(0, 0, 0);
    for (int _ = 0; _ < num_samples_per_pixel; ++_) {
        Point px = pixel;
        if (_ > 0)
            px += delta_w*get_random_double(d_states, r, -0.5, 0.5) + delta_h*get_random_double(d_states, r, -0.5, 0.5);
        const Point ray_direction = px - camera_position;
        s += ray_trajectory(px, ray_direction, star_map, sm_image_width, sm_image_height, disk_map, disk_map_x, disk_map_y, disk_map_z, time, create_disk, probty_constant, nullptr);
    }
    image[r] = s / num_samples_per_pixel;
}

struct Image {
    vector<Point> image;
    unsigned int image_width, image_height;

    inline Image(const vector<Point>& image, int image_width, int image_height):
    image(image), image_width(image_width), image_height(image_height) {}

    inline void read_png(const string infilepath) {
        std::vector<unsigned char> png;

        lodepng::decode(png, image_width, image_height, infilepath.c_str(), LCT_RGB, 8);
        image.clear();
        image.reserve(image_width * image_height);
        
        for (size_t i = 0; i < png.size(); i += 3) {
            image.push_back({
                (double) png[i]     / MAX_COLOR,
                (double) png[i + 1] / MAX_COLOR,
                (double) png[i + 2] / MAX_COLOR
            });
        }
    }

    inline Image(const string& infilepath) {
        read_png(infilepath);
    }

    inline void write_png(const string& outfilepath) const {
        vector<unsigned char> png(image_width*image_height*3);
        for (size_t r = 0; r < image.size(); ++r) {
            png[3*r    ] = clamp(int(image[r][0] * MAX_COLOR), 0, 255);
            png[3*r + 1] = clamp(int(image[r][1] * MAX_COLOR), 0, 255);
            png[3*r + 2] = clamp(int(image[r][2] * MAX_COLOR), 0, 255);
        }
        unsigned error = lodepng::encode(outfilepath.c_str(), png, image_width, image_height, LCT_RGB, 8);
    }
};

struct Disk3D {
    int x, y, z;
    vector<double> data;
};

Disk3D read_disk3d_from_file(const string& filepath) {
    ifstream fin(filepath);
    Disk3D v;
    fin >> v.x >> v.y >> v.z;
    const int total_vals = v.x * v.y * v.z;
    v.data = vector<double>(total_vals);
    for (int i = 0; i < total_vals; ++i)
        fin >> v.data[i];
    return v;
}

struct CameraSettings {
    Point camera_position = Point(0, UNIVERSE_RADIUS/2 * sin(3*pi/180), UNIVERSE_RADIUS/2 * cos(3*pi/180));
    Point look_to = Point(0, 0, 0);
    double focal_length = 1;
    double view_angle = atan(ACCRETION_DISK_OUTER_RADIUS / (16. / 9) * 5 / 2 / camera_position.length())*2;
    Point universe_up = Point(0, 1, 0);
    double time = 0;
    int num_samples_per_pixel = 1;
    string image_filepath = "../data/example.png";
    bool create_disk = true;
    double probty_constant = 1;
    double black_hole_mass = 1;
};

class Camera {
    // set
    const double aspect_ratio = 16./9;
    const int image_height = 200;
    Point camera_position;
    Point look_to;
    double focal_length;
    double view_angle;
    Point universe_up;
    double time;
    int num_samples_per_pixel;

    public:
    Camera() {}
    Camera(double aspect_ratio, int image_height):
    aspect_ratio(aspect_ratio), image_height(image_height) {}
    private:
    
    // calculate
    const int image_width = aspect_ratio*image_height;
    double viewport_height;
    double viewport_width;
    Point z;
    Point x;
    Point y;
    Point vector_w;
    Point vector_h;
    Point delta_w;
    Point delta_h;
    Point top_left_position;
    Point pixel00_position;

    inline void initialize() {
        viewport_height = 2 * tan(view_angle / 2) * focal_length;
        viewport_width = viewport_height * (double) image_width / image_height;
        z = normalize(camera_position - look_to);
        x = normalize(cross(universe_up, z));
        y = cross(z, x);
        vector_w = viewport_width*x;
        vector_h = -viewport_height*y;
        delta_w = vector_w / image_width;
        delta_h = vector_h / image_height;
        top_left_position = camera_position - z * focal_length - (vector_w + vector_h) / 2;
        pixel00_position = top_left_position + (delta_w + delta_h) / 2;
    }

    inline void initialize(const CameraSettings& camset) {
        camera_position = camset.camera_position;
        look_to = camset.look_to;
        focal_length = camset.focal_length;
        view_angle = camset.view_angle;
        universe_up = camset.universe_up;
        time = camset.time;
        num_samples_per_pixel = camset.num_samples_per_pixel;

        initialize();
    }

    vector<CameraSettings> drop_for_existing_filename(const vector<CameraSettings>& camset) {
        vector<CameraSettings> ans;
        for (const auto& cs: camset) {
            if (!filePresent(cs.image_filepath))
                ans.push_back(cs);
        }
        return ans;
    }

    public:

    inline void render(const Image& starmap, const Disk3D& diskmap, vector<CameraSettings> camset) {
        initialize();
        const int total_pixels = image_width * image_height;
        const int BLOCKSIZE = 64;
        const int NUMBLOCKS = (total_pixels + BLOCKSIZE - 1) / BLOCKSIZE;

        curandState* d_states;
        cudaMalloc(&d_states, total_pixels * sizeof(curandState));
        setup_curand_kernel<<<NUMBLOCKS, BLOCKSIZE>>>(d_states, 1234ULL, total_pixels);
        cudaDeviceSynchronize();

        Point *image_d, *starmap_d;
        double *diskmap_d;
        cudaMalloc(&image_d, total_pixels * sizeof(Point));
        cudaMalloc(&starmap_d, starmap.image.size() * sizeof(Point));
        cudaMalloc(&diskmap_d, diskmap.data.size() * sizeof(double));
        cudaMemcpy(starmap_d, starmap.image.data(), starmap.image.size() * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(diskmap_d, diskmap.data.data(), diskmap.data.size() * sizeof(double), cudaMemcpyHostToDevice);

        Image output_image(vector<Point>(total_pixels), image_width, image_height);
        int frame = 0;
        const auto start = chrono::steady_clock::now();
        for (const CameraSettings& cs: camset) {
            set_black_hole_constants(cs.black_hole_mass);
            initialize(cs);
            kernel<<<NUMBLOCKS, BLOCKSIZE>>>(
                image_d,
                image_width,
                image_height,
                pixel00_position,
                delta_w,
                delta_h,
                camera_position,
                starmap_d,
                starmap.image_width,
                starmap.image_height,
                diskmap_d,
                diskmap.x,
                diskmap.y,
                diskmap.z,
                time,
                d_states,
                num_samples_per_pixel,
                cs.create_disk,
                cs.probty_constant
            );
            cudaDeviceSynchronize();
            cudaMemcpy(output_image.image.data(), image_d, total_pixels * sizeof(Point), cudaMemcpyDeviceToHost);
            output_image.write_png(cs.image_filepath);
            frame += 1;
            const auto end = chrono::steady_clock::now();
            const auto duration = (double) chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000;
            const double avg_time_per_iteration = duration / frame;
            const double est_time_left = (camset.size() - frame) * avg_time_per_iteration;
            cout << "\r" << frame << " / " << camset.size() << ", elapsed = " << duration << " s, time/frame = " << avg_time_per_iteration << " s, left = " << est_time_left << " s          ";
            cout.flush();
        }
        cout << '\n';

        cudaFree(image_d);
        cudaFree(starmap_d);
        cudaFree(diskmap_d);
        cudaFree(d_states);
    }
};

#endif
