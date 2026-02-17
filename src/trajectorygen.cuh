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

#ifndef TRAJECTORYGEN_CUH
#define TRAJECTORYGEN_CUH

#include <utilities.cuh>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

/**
 * the following functions generate ray trajectories which can be plotted using `plot_trajectory.py` script
 */

inline void trajectory_gen_01() {
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

inline void trajectory_gen_02() {
    int count = 0;
    const double R = 4*SCHWARZSCHILD_RADIUS;
    for (double ang = 0; ang < 2*pi; ang += 2*pi/12) {
        vector<Point> trajectory(60000, Point(0,0,0));
        Point* sm = new Point(0 ,0, 0);
        double* dm = new double(0);
        const Point O(0, 0, 10);
        const Point D = Point(R*cos(ang), R*sin(ang), 0) - O;
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

inline void trajectory_gen_03() {
    int count = 0;
    const double R = 4*SCHWARZSCHILD_RADIUS;
    for (double x = -8; x <= 8; x += 1) {
        vector<Point> trajectory(60000, Point(0,0,0));
        Point* sm = new Point(0 ,0, 0);
        double* dm = new double(0);
        const Point O(0, 0, 10);
        const Point D = Point(x, 0, 0) - O;
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
