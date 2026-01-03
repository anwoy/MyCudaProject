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

#ifndef TESTS_CUH
#define TESTS_CUH

#include <utilities.cuh>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

__global__ void test_cuda_random(curandState* d_states, double* image, const int num_pixels) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_pixels)
        return;
    const double random_val = get_random_double(d_states, r);
    image[r] = random_val;
}

inline void tester01() {
    const int num_pixels = 1e6;
    curandState* d_state;
    cudaMalloc(&d_state, num_pixels * sizeof(curandState));
    const int BLOCKSIZE = 64;
    const int NUMBLOCKS = (num_pixels + BLOCKSIZE - 1) / BLOCKSIZE;
    setup_curand_kernel<<<NUMBLOCKS, BLOCKSIZE>>>(d_state, 1234ULL, num_pixels);
    cudaDeviceSynchronize();
    
    cout << "\n";
    double* d_image;
    cudaMalloc(&d_image, num_pixels * sizeof(double));
    vector<double> h_image(num_pixels);
    for (int i = 0; i < 10; ++i) {
        test_cuda_random<<<NUMBLOCKS, BLOCKSIZE>>>(d_state, d_image, num_pixels);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image.data(), d_image, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);
        ofstream fout(to_string(i) + ".txt");
        for (double v: h_image)
            fout << v << " ";
        cout << '\r' << i << " / " << 10 << "                ";
        cout.flush();
    }
    
    cudaFree(d_image);
    cudaFree(d_state);
}

inline void tester02() {
    unsigned width = 256, height = 256;
    // The image data is stored as a vector of unsigned chars,
    // 3 bytes per pixel (R, G, B)
    std::vector<unsigned char> image;
    image.resize(width * height * 3);

    // Generate a simple checkered image
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            std::size_t index = 3 * (y * width + x);
            // Red channel
            image[index + 0] = (x + y) % 2 * 255;
            // Green channel
            image[index + 1] = 0;
            // Blue channel
            image[index + 2] = 0;
        }
    }

    lodepng_encode24_file(string("../data/test.png").data(), image.data(), width, height);
}

inline void tester03() {
    int count = 0;
    for (double x = -8.1; x <= 0.1; x += 1) {
        vector<Point> trajectory(60000, Point(0,0,0));
        Point* sm = new Point(0 ,0, 0);
        double* dm = new double(0);
        ray_trajectory(
            Point(x,0,10),
            Point(0, 0, -1),
            sm,1,1,
            dm,1,1,1,
            0,
            false,
            1,
            trajectory.data()
        );
        ofstream fout("../data/trajectory_" + to_string(count) + ".csv");
        count += 1;
        fout << "x,y,z\n";
        for (const Point& p: trajectory) {
            if (p.get_x() == 0 && p.get_y() == 0 && p.get_z() == 0)
                break;
            fout << p << '\n';
        }
        free(sm);
        free(dm);
    }
}

inline void tester04() {
    int count = 0;
    const double R = 4*SCHWARZSCHILD_RADIUS;
    for (double ang = 0; ang < 2*pi; ang += 2*pi/12) {
        vector<Point> trajectory(60000, Point(0,0,0));
        Point* sm = new Point(0 ,0, 0);
        double* dm = new double(0);
        const Point O(0, 0, 10);
        const Point D = Point(R*cos(ang), R*sin(ang), 0) - O;
        cout << O << " : " << D << '\n';
        ray_trajectory(
            O,
            D,
            sm,1,1,
            dm,1,1,1,
            0,
            false,
            1,
            trajectory.data()
        );
        ofstream fout("../data/trajectory_" + to_string(count) + ".csv");
        count += 1;
        fout << "x,y,z\n";
        for (const Point& p: trajectory) {
            if (p.get_x() == 0 && p.get_y() == 0 && p.get_z() == 0)
                break;
            fout << p << '\n';
        }
        free(sm);
        free(dm);
    }
}

inline void tester05() {
    int count = 0;
    const double R = 4*SCHWARZSCHILD_RADIUS;
    for (double x = -8; x <= 8; x += 1) {
        vector<Point> trajectory(60000, Point(0,0,0));
        Point* sm = new Point(0 ,0, 0);
        double* dm = new double(0);
        const Point O(0, 0, 10);
        const Point D = Point(x, 0, 0) - O;
        cout << O << " : " << D << '\n';
        ray_trajectory(
            O,
            D,
            sm,1,1,
            dm,1,1,1,
            0,
            false,
            1,
            trajectory.data()
        );
        ofstream fout("../data/trajectory_" + to_string(count) + ".csv");
        count += 1;
        fout << "x,y,z\n";
        for (const Point& p: trajectory) {
            if (p.get_x() == 0 && p.get_y() == 0 && p.get_z() == 0)
                break;
            fout << p << '\n';
        }
        free(sm);
        free(dm);
    }
}

#endif
