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
 */

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <curand_kernel.h>
#include <fstream>

using namespace std;

/**
 * Initializes cuRAND states for a 1D array of pixels.
 * @param states Pointer to the global memory array of curandState objects.
 * @param seed The base seed used to initialize the generator.
 * @param total_pixels total pixels in a rendered frame
 */
__global__ void setup_curand_kernel(curandState* states, unsigned long seed, int total_pixels) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < total_pixels) {
        curand_init(seed, r, 0, &states[r]);
    }
}

/**
 * Generates a pseudo-random double in the range (0.0, 1.0].
 * @param global_state Pointer to the array of pre-initialized curandState objects.
 * @param r The 1D global index for this thread.
 * @return random double number in the range (0.0, 1.0].
 */
__device__ inline double get_random_double(curandState* global_state, const int r) {
    curandState localState = global_state[r];
    double val = curand_uniform_double(&localState);
    global_state[r] = localState;
    return val;
}

/**
 * generates a random double number between `low` and `high`
 * @param global_state Pointer to the array of pre-initialized curandState objects.
 * @param r The 1D global index for this thread.
 * @param low lower bound
 * @param high upper bound
 * @return random double number between `low` and `high`
 */
__device__ inline double get_random_double(curandState* global_state, const int r, const double low, const double high) {
    const double val = get_random_double(global_state, r);
    return low + val*(high - low);
}

/**
 * clamps a value to stay between `lower` and `upper`
 * @param val input value
 * @param lower lower bound
 * @param upper upper bound
 * @return clamped value
 */
__device__ __host__ inline int clamp(const int val, const int lower, const int upper) {
    return min(max(lower, val), upper);
}

/**
 * clamp version for double
 */
__device__ __host__ inline int fclamp(const double val, const double lower, const double upper) {
    return min(max(lower, val), upper);
}

/**
 * outputs linear interpolated value at `t`
 * @verbatim
            `upper_val` ---------------- /
                                      /   |
            `output` -------------- /     |
                                 / |      |
            `lower_val` ----  /    |      |
                             |     |      |
                         `lower`  `t`  `upper`
 * @endverbatim
 */
__device__ __host__ inline double linear_interpolation(const double t, const double lower, const double upper, double lower_val, double upper_val) {
    return (t - lower) / (upper - lower)*(upper_val - lower_val) + lower_val;
}

/**
 * similar to `linear_interpolation`, with an extra steepness parameter
 */
inline double smooth_step(double t, const double lower, const double upper, double lower_val, double upper_val, double steep) {
    t = fclamp(t, lower, upper);
    const double temp = 2 * (t - lower) / (upper - lower) - 1;
    const double s = (tanh(steep*temp) / tanh(steep) + 1) / 2;
    return s * (upper_val - lower_val) + lower_val;
}

/**
 * `linear_interpolation` version for 3D points
 */
__device__ __host__ inline Point linear_interpolation(const double t, const double lower, const double upper, const Point& lower_val, const Point& upper_val) {
    return (t - lower) / (upper - lower)*(upper_val - lower_val) + lower_val;
}

#endif
