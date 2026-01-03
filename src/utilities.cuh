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

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <curand_kernel.h>
#include <fstream>

using namespace std;

__global__ void setup_curand_kernel(curandState* states, unsigned long seed, int total_pixels) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < total_pixels) {
        curand_init(seed, r, 0, &states[r]);
    }
}

__device__ inline double get_random_double(curandState* global_state, const int r) {
    curandState localState = global_state[r];
    double val = curand_uniform_double(&localState);
    global_state[r] = localState;
    return val;
}

__device__ inline double get_random_double(curandState* global_state, const int r, const double low, const double high) {
    const double val = get_random_double(global_state, r);
    return low + val*(high - low);
}

__device__ inline Point get_random_unit_vector(curandState* global_state, const int r) {
    while (true) {
        const Point p(
            get_random_double(global_state, r, -1, 1),
            get_random_double(global_state, r, -1, 1),
            get_random_double(global_state, r, -1, 1)
        );
        if (p.length_squared() < 1)
            return normalize(p);
    }
}

__device__ __host__ inline int clamp(const int val, const int lower, const int upper) {
    return min(max(lower, val), upper);
}

__device__ __host__ inline int fclamp(const double val, const double lower, const double upper) {
    return min(max(lower, val), upper);
}

__device__ __host__ inline double linear_interpolation(const double t, const double lower, const double upper, double lower_val, double upper_val) {
    return (t - lower) / (upper - lower)*(upper_val - lower_val) + lower_val;
}

__device__ __host__ inline double linear_interpolation(double t, const double lower, const double upper, double lower_val, double upper_val, const double power) {
    t = fclamp(t, lower, upper);
    return pow((t - lower) / (upper - lower), power)*(upper_val - lower_val) + lower_val;
}

inline double smooth_step(double t, const double lower, const double upper, double lower_val, double upper_val, double steep) {
    t = fclamp(t, lower, upper);
    const double temp = 2 * (t - lower) / (upper - lower) - 1;
    const double s = (tanh(steep*temp) / tanh(steep) + 1) / 2;
    return s * (upper_val - lower_val) + lower_val;
}

__device__ __host__ inline Point linear_interpolation(const double t, const double lower, const double upper, const Point& lower_val, const Point& upper_val) {
    return (t - lower) / (upper - lower)*(upper_val - lower_val) + lower_val;
}

inline bool filePresent(const std::string& filename) {
    ifstream file(filename);
    return file.good(); // or return file.is_open();
}

#endif
