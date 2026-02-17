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

#ifndef POINT_CUH
#define POINT_CUH

using namespace std;

/**
 * A general 3D vector and associated operations.
 * This can be used to represent any general 3D state
 * such as a point in 3D space or RGB color.
 */
class Point {
    double x, y, z;

public:
    __device__ __host__ inline Point() {}

    __device__ __host__ inline Point(double x, double y, double z) : x(x), y(y), z(z) {}

    __device__ __host__ inline double get_x() const { return x; }
    __device__ __host__ inline double get_y() const { return y; }
    __device__ __host__ inline double get_z() const { return z; }

    __device__ __host__ inline double operator[](size_t i) const {
        switch (i)
        {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        case 2:
            return z;
            break;
        default:
            return x;
            break;
        }
    }

    __device__ __host__ inline double& operator[](size_t i) {
        switch (i)
        {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        case 2:
            return z;
            break;
        default:
            return x;
            break;
        }
    }

    __device__ __host__ inline Point& operator+=(const Point& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __device__ __host__ inline Point& operator-=(const Point& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    __device__ __host__ inline Point& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    __device__ __host__ inline Point& operator*=(const Point& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    __device__ __host__ inline Point& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    __device__ __host__ inline double length_squared() const {
        return x*x + y*y + z*z;
    }

    __device__ __host__ inline double length() const { return sqrt(fmax(0., length_squared())); }
};

inline ostream& operator<<(ostream& out, const Point& p) {
    out << p[0] << ',' << p[1] << ',' << p[2];
    return out;
}

__device__ __host__ inline Point operator+(Point lhs, const Point& rhs) {
    lhs += rhs;
    return lhs;
}

__device__ __host__ inline Point operator-(Point lhs, const Point& rhs) {
    lhs -= rhs;
    return lhs;
}

__device__ __host__ inline Point operator*(Point p, double scalar) {
    p *= scalar;
    return p;
}

__device__ __host__ inline Point operator*(Point lhs, const Point& rhs) {
    lhs *= rhs;
    return lhs;
}

__device__ __host__ inline Point operator*(double scalar, Point p) {
    return p * scalar;
}

__device__ __host__ inline Point operator-(const Point& p) {
    return -1*p;
}

__device__ __host__ inline Point operator/(Point lhs, double scalar) {
    lhs /= scalar;
    return lhs;
}

__device__ __host__ inline Point normalize(const Point& p) {
    return p / p.length();
}

__device__ __host__ inline double dot(const Point& a, const Point& b) {
    double result = 0;
    for (size_t i = 0; i < 3; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

__device__ __host__ inline Point cross(const Point& a, const Point& b) {
    return Point(
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0])
    );
}

#endif
