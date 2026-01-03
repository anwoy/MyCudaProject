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

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

using namespace std;

__managed__ Point INNER_COLOR;
__managed__ Point OUTER_COLOR;
__managed__ Point BACKGROUND_COLOR;

__managed__ double OMEGA_CONSTANT;

const double pi = 3.141592653589793;
const int MAX_COLOR = 255;
const double SPEED_OF_LIGHT = 1;
const double GRAVITATIONAL_CONSTANT = 1;
const double UNIVERSE_RADIUS = 60;

__managed__ double BLACK_HOLE_MASS;
__managed__ double SCHWARZSCHILD_RADIUS;
__managed__ double ACCRETION_DISK_INNER_RADIUS;
__managed__ double ACCRETION_DISK_OUTER_RADIUS;
__managed__ double ACCRETION_DISK_INNER_THICKNESS;
__managed__ double ACCRETION_DISK_OUTER_THICKNESS;

inline __device__ __host__ void set_black_hole_constants(double mass) {
    BLACK_HOLE_MASS = mass;
    SCHWARZSCHILD_RADIUS = 2*GRAVITATIONAL_CONSTANT*BLACK_HOLE_MASS/SPEED_OF_LIGHT/SPEED_OF_LIGHT;
    ACCRETION_DISK_INNER_RADIUS = SCHWARZSCHILD_RADIUS*3;
    ACCRETION_DISK_OUTER_RADIUS = SCHWARZSCHILD_RADIUS*15;
    ACCRETION_DISK_INNER_THICKNESS = ACCRETION_DISK_INNER_RADIUS*0.05;
    ACCRETION_DISK_OUTER_THICKNESS = ACCRETION_DISK_OUTER_RADIUS*0.05;
    OMEGA_CONSTANT = 2*2*pi*sqrt(ACCRETION_DISK_INNER_RADIUS*ACCRETION_DISK_INNER_RADIUS*ACCRETION_DISK_INNER_RADIUS);
}



#endif
