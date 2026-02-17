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

#ifndef RAY_CUH
#define RAY_CUH

#include <point.cuh>
#include <constants.cuh>
#include <utilities.cuh>

using namespace std;

/**
 * A state is represented as a 3D vector [u, du/dphi, phi], this function calculates the derivative as a function of the state
 * @param state input state
 * @return the derivative given the input
 */
__device__ __host__ inline Point derivative(const Point& state) {
    const double y1 = state[0], y2 = state[1];
    
    return Point {
        y2,
        SCHWARZSCHILD_RADIUS*3/2*y1*y1 - y1,
        1
    };
}

/**
 * A struct to save initial conditions, and the x, y, and z directions of the frame in which the path equation is solved
 */
struct InitialConditions {
    const Point x, y, z, initial_state;

    __device__ __host__ inline InitialConditions(const Point& x, const Point& y, const Point& z, const Point& initial_state):
    x(x), y(y), z(z), initial_state(initial_state) {}
};

/**
 * Returns an InitialConditions object given the initial point of origin and direction of propagation of a photon.
 * @param origin position of the photon
 * @param direction direction of propagation of the photon
 * @return Initial state and transformed frame bundled in an `InitialConditions` object
 */
__device__ __host__ inline InitialConditions initial_conditions_calc(const Point& origin, const Point& direction) {
    const Point x = normalize(direction);
    const Point z = normalize(cross(direction, origin));
    const Point y = cross(z, x);
    const double _x = dot(x, origin), _y = dot(y, origin);
    const double phi_0 = atan2(_y, _x);
    const double r_0 = origin.length();
    const double u_0 = 1 / r_0;
    const double p = dot(x, direction), q = dot(y, direction);
    const double R = (p*_x + q*_y) / r_0;
    const double T = (q*_x - p*_y) / r_0;
    const Point initial_state {
        u_0,
        -R / r_0 / T,
        phi_0
    };
    return InitialConditions(x, y, z, initial_state);
}

/**
 * saves a position in 3D space as well as the distance from origin of said point
 */
struct Position {
    double r;
    Point position;

    __device__ __host__ inline Position(const double r, const Point& position):
    r(r), position(position) {}
};

/**
 * derives position from the state, using the x, y, z directions of the frame in which the path equation is solved
 * @param state a 3D vector representing [u, du/dphi, phi]
 * @param it an `InitialConditions` object containing the transformed frame
 */
__device__ __host__ inline Position get_position(const Point& state, const InitialConditions& it) {
    const double r = 1 / state[0];
    const double angle = state[2];
    const Point position = r*cos(angle)*it.x + r*sin(angle)*it.y;
    return Position(r, position);
}

/**
 * assuming position `p` is within the universe boundary, and position `next_p` is outside, this function returns the point on the sphere representing the universe boundary
 * where the photon passes.
 * @param p point inside universe boundary
 * @param next_p point outside universe boundary
 * @return point on the universe boundary where ray hits
 */
__device__ __host__ inline Point get_hitpoint_on_universe_boundary(const Point& p, const Point& next_p) {
    const Point &p1 = p, &p2 = next_p;
    const Point d = p2 - p1;
    const double d_l_sq = d.length_squared();
    const double b = dot(p1, d) / d_l_sq;
    const double c = (p1.length_squared() - UNIVERSE_RADIUS*UNIVERSE_RADIUS) / d_l_sq;
    const double disc = b*b - c;
    const double t = sqrt(fmax(0., disc)) - b;
    return p1 + t*d;
}

/**
 * convert hit point on universe boundary to latitude and longitude and save in this struct
 */
struct LatLong {
    double latitude, longitude;
    __device__ __host__ inline LatLong(double latitude, double longitude):
    latitude(latitude), longitude(longitude) {}
};

/**
 * convert hit point on universe boundary to latitude and longitude
 * @param p point on the universe boundary where ray hits
 * @return latitude and longitude bundled in a `LatLong` object
 */
__device__ __host__ inline LatLong get_lat_long_from_xyz(const Point& p) {
    const double r = p.length();
    const double latitude = asin(p.get_z() / r);
    const double longitude = atan2(p.get_y(), p.get_x());
    return LatLong(latitude, longitude);
}

/**
 * RK4 step to compute next state from present state
 * @param state a 3D vector representing [u, du/dphi, phi]
 * @param step_size step size used in the RK4 iterations
 * @return next state
 */
__device__ __host__ inline Point next_state_calc(const Point& state, const double step_size) {
    const Point k1 = derivative(state);
    const Point k2 = derivative(state + step_size / 2 * k1);
    const Point k3 = derivative(state + step_size / 2 * k2);
    const Point k4 = derivative(state + step_size * k3);
    return state + step_size / 6 * (k1 + 2*k2 + 2*k3 + k4);
}

/**
 * This struct is returned from the function which checks if a photon is within the accretion disk
 */
struct DiskInfo {
    bool condition;
    double thickness, l;
};

/**
 * checks if a photon at position `p` is within the accretion disk
 * @param p 3D vector representing a point in space
 * @return metadata bundled in a `DiskInfo` object
 */
__device__ __host__ inline DiskInfo check_inside_disk(const Point& p) {
    DiskInfo ans;
    const double x = p.get_x(), y = p.get_y(), z = p.get_z();
    const double l = sqrt(x*x + z*z);
    ans.l = l;
    if (ACCRETION_DISK_INNER_RADIUS < l && l < ACCRETION_DISK_OUTER_RADIUS) {
        const double thickness = linear_interpolation(l, ACCRETION_DISK_INNER_RADIUS, ACCRETION_DISK_OUTER_RADIUS, ACCRETION_DISK_INNER_THICKNESS, ACCRETION_DISK_OUTER_THICKNESS);
        ans.thickness = thickness;
        ans.condition = (-thickness / 2 < y && y < thickness / 2);
        return ans;
    }
    ans.condition = false;
    return ans;
}

/**
 * function describing density of accretion disk
 * @param y distance from equitorial plane
 * @param thickness thickness of the disk at that location
 * @return density
 */
__device__ __host__ inline double vertical_dependance(double y, double thickness) {
    return exp(-y*y/thickness/thickness/2);
}

/**
 * function describing density of accretion disk
 * @param l distance of the point from the origin on the equitorial plane
 * @return density
 */
__device__ __host__ inline double radial_dependance(double l) {
    return 1./l/sqrt(l)*(1 - sqrt(ACCRETION_DISK_INNER_RADIUS / l))*linear_interpolation(l, ACCRETION_DISK_INNER_RADIUS, ACCRETION_DISK_OUTER_RADIUS, 1, 0);
}

/**
 * Computes ray trajectory
 * @param origin start poition of the photon
 * @param direction direction of the photon
 * @param star_map star map in plate carrée format
 * @param sm_image_width width of the star map
 * @param sm_image_height height of the star map
 * @param disk_map Perlin noise map for the accretion disk
 * @param disk_map_x width of the disk map
 * @param disk_map_y height of the disk map
 * @param disk_map_z thickness of the disk map
 * @param time used rotate accretion disk
 * @param create_disk set to false to not render accretion disk
 * @param disk_density_modulator used to modulate density
 * @param trajectory save trajectory for debugging
 * @return color to ascribe to the pixel from where ray originated
 */
__device__ __host__ inline Point ray_trajectory(
    const Point& origin,
    const Point& direction,
    Point* star_map,
    const int sm_image_width,
    const int sm_image_height,
    double* disk_map,
    const int disk_map_x,
    const int disk_map_y,
    const int disk_map_z,
    const double time,
    const bool create_disk,
    const double disk_density_modulator,
    Point* trajectory
) {
    const InitialConditions initial_conditions = initial_conditions_calc(origin, direction);
    double step_size = 1./20;
    const double march_step_size = 1e-2;
    Point state = initial_conditions.initial_state;
    Position position = get_position(state, initial_conditions);
    Point color(0, 0, 0);
    Point transmission(1,1,1);
    for (int i = 0; i < 60000; ++i) {
#ifndef __CUDA_ARCH__
        trajectory[i] = position.position;
#endif
        step_size = min(max(1e-3/.2*fabs(state[2] - pi), 1e-3), 1./20);
        const Point next_state = next_state_calc(state, -step_size);
        const Position next_position = get_position(next_state, initial_conditions);
        int num_steps = 1;
        const double min_length = fmin(next_position.position.length(), position.position.length());
        if (check_inside_disk(position.position).condition || check_inside_disk(next_position.position).condition)
            num_steps = clamp((next_position.position - position.position).length() / march_step_size, 1, 50);
        const Point increment = (next_position.position - position.position) / num_steps;
        const double increment_length = increment.length();
        Point _p = position.position;
        for (int j = 1; j < num_steps + 1; ++j) {
            const Point _next_p = position.position + j*increment;
            if (_next_p.length() < SCHWARZSCHILD_RADIUS)
                return color + BACKGROUND_COLOR*transmission;
            if (_p.length() < UNIVERSE_RADIUS && _next_p.length() > UNIVERSE_RADIUS) {
                const Point pos_on_universe = get_hitpoint_on_universe_boundary(_p, _next_p);
                const LatLong latlong = get_lat_long_from_xyz(pos_on_universe);
                const int i = (latlong.longitude + pi) / 2 / pi * sm_image_width;
                const int j = (latlong.latitude + pi / 2) / pi * sm_image_height;
                const int r = min(j * sm_image_width + i, sm_image_width * sm_image_height);
                return color + star_map[r]*transmission;
            }
            if (create_disk) {
                const DiskInfo disk_info = check_inside_disk(_next_p);
                if (disk_info.condition) {
                    const double angle = atan2(_next_p.get_z(), _next_p.get_x());
                    const double omega = OMEGA_CONSTANT / sqrt(disk_info.l*disk_info.l*disk_info.l);
                    const int _i = fmod(angle + time * omega + pi, 2*pi) / 2 / pi * disk_map_x;
                    const int _j = linear_interpolation(disk_info.l, ACCRETION_DISK_INNER_RADIUS, ACCRETION_DISK_OUTER_RADIUS, 0, disk_map_y);
                    const int _k = linear_interpolation(_next_p.get_y(), -disk_info.thickness/2, disk_info.thickness/2, 0, disk_map_z);
                    const int r = min(_k + disk_map_z*_j + disk_map_z*disk_map_y*_i, disk_map_x*disk_map_y*disk_map_z - 1);

                    const double density = disk_map[r]*vertical_dependance(_next_p.get_y(), disk_info.thickness/10)*radial_dependance(disk_info.l)*5000*disk_density_modulator;
                    const double luminosity = 1./disk_info.l*5;
                    const Point disk_color = linear_interpolation(disk_info.l, ACCRETION_DISK_INNER_RADIUS, ACCRETION_DISK_OUTER_RADIUS, INNER_COLOR, OUTER_COLOR);
                    color += transmission * density * luminosity * disk_color * increment_length;
                    transmission *= exp(-density * increment_length); // absorption
                }
            }
            _p = _next_p;
        }
        state = move(next_state);
        position = move(next_position);
    }
    return color + BACKGROUND_COLOR*transmission;
}

#endif
