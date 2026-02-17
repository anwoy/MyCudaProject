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

#include <iostream>

#include <camera.cuh>
#include <trajectorygen.cuh>

using namespace std;

/**
 * functions clip_gen_01 through clip_gen_07 generate separate
 * movie clips and save the frames as png images
 */

 /**
  * generates frames of a movie clip as png images. Same format for other clip_gen functions as well
  * @param num_frames number of frames to generate
  * @param num_samples_per_pixel number of monte carlo samples to generate per pixel for anti aliasing
  * @return vector of `CameraSettings` objects
  */
vector<CameraSettings> clip_gen_01(const int num_frames, const int num_samples_per_pixel) {
    const Point up0(0.2, 1, 0), up1(-0.2, 1, 0);
    const double latitude0 = pi/8, latitude1 = -pi/8;
    const double longitude0 = pi + pi/2, longitude1 = pi/2;
    const double l0 = 30*SCHWARZSCHILD_RADIUS, l1 = 20*SCHWARZSCHILD_RADIUS;

    CameraSettings cs;
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*10 / 2 / l0)*2;
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        cs.universe_up = linear_interpolation(i, 0, num_frames, up0, up1);
        const double latitude = linear_interpolation(i, 0, num_frames, latitude0, latitude1);
        const double longitiude = linear_interpolation(i, 0, num_frames, longitude0, longitude1);
        const double l = linear_interpolation(i, 0, num_frames, l0, l1);
        cs.camera_position = Point(
            l * cos(latitude) * sin(longitiude),
            l * sin(latitude),
            l * cos(latitude) * cos(longitiude)
        );
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.image_filepath = "../data/clip01_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}

vector<CameraSettings> clip_gen_02(const int num_frames, const int num_samples_per_pixel) {
    const double l = SCHWARZSCHILD_RADIUS*8;
    const double ang = 6; // degrees
    const Point cp0(0, l*sin(ang*pi/180), l*cos(ang*pi/180)), cp1(0, -l*sin(ang*pi/180), l*cos(ang*pi/180));

    CameraSettings cs;
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.look_to = Point(SCHWARZSCHILD_RADIUS*5.5, 0, 0);
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*10 / 2 / cs.camera_position.length())*2 / 3;
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        cs.camera_position = linear_interpolation(i, 0, num_frames, cp0, cp1);
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.image_filepath = "../data/clip02_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}

vector<CameraSettings> clip_gen_03(const int num_frames, const int num_samples_per_pixel) {
    const Point cp0(
        ACCRETION_DISK_INNER_RADIUS,
        -ACCRETION_DISK_INNER_THICKNESS*2,
        ACCRETION_DISK_OUTER_RADIUS
    ), cp1(
        ACCRETION_DISK_INNER_RADIUS,
        ACCRETION_DISK_INNER_THICKNESS*2,
        SCHWARZSCHILD_RADIUS*8
    );

    CameraSettings cs;
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*10 / cs.camera_position.length()/2)*2 / 3;
    cs.look_to = Point(2*ACCRETION_DISK_INNER_RADIUS, 0, 0);
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        cs.camera_position = linear_interpolation(i, 0, num_frames, cp0, cp1);
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.image_filepath = "../data/clip03_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}

vector<CameraSettings> clip_gen_04(const int num_frames, const int num_samples_per_pixel) {
    const double latitude0 = pi/8, latitude1 = pi/16;
    const double longitude0 = pi * 0.9, longitude1 = pi * 0.7;
    const double l = 25*SCHWARZSCHILD_RADIUS;

    CameraSettings cs;
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*10 / 2 / l)*2;
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        const double latitude = linear_interpolation(i, 0, num_frames, latitude0, latitude1);
        const double longitiude = linear_interpolation(i, 0, num_frames, longitude0, longitude1);
        cs.camera_position = Point(
            l * cos(latitude) * sin(longitiude),
            l * sin(latitude),
            l * cos(latitude) * cos(longitiude)
        );
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.disk_density_modulator = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.image_filepath = "../data/clip04_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}

vector<CameraSettings> clip_gen_05(const int num_frames, const int num_samples_per_pixel) {
    const double latitude0 = pi/8, latitude1 = pi/16;
    const double longitude0 = pi, longitude1 = pi * 0.8;
    const double l = 25*SCHWARZSCHILD_RADIUS;

    CameraSettings cs;
    cs.create_disk = false;
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*7 / 2 / l)*2;
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        const double latitude = linear_interpolation(i, 0, num_frames, latitude0, latitude1);
        const double longitiude = linear_interpolation(i, 0, num_frames, longitude0, longitude1);
        cs.camera_position = Point(
            l * cos(latitude) * sin(longitiude),
            l * sin(latitude),
            l * cos(latitude) * cos(longitiude)
        );
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.black_hole_mass = smooth_step(i, 0, num_frames, 0, 1, 10);
        cs.image_filepath = "../data/clip05_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}

vector<CameraSettings> clip_gen_06(const int num_frames, const int num_samples_per_pixel) {
    const Point cp0(
        ACCRETION_DISK_OUTER_RADIUS,
        ACCRETION_DISK_OUTER_RADIUS,
        ACCRETION_DISK_OUTER_RADIUS
    ), cp1(
        0,
        -ACCRETION_DISK_OUTER_THICKNESS*3,
        ACCRETION_DISK_OUTER_RADIUS/2
    );
    const Point lt0(0,0,0), lt1(
        SCHWARZSCHILD_RADIUS*7,
        0,
        0
    );
    const Point up0(-1, 1, 0), up1(0, 1, 0);

    CameraSettings cs;
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*10 / 2 / cp0.length())*2;
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        cs.camera_position = linear_interpolation(i, 0, num_frames, cp0, cp1);
        cs.universe_up = linear_interpolation(i, 0, num_frames, up0, up1);
        cs.look_to = linear_interpolation(i, 0, num_frames, lt0, lt1);
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.image_filepath = "../data/clip06_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}

vector<CameraSettings> clip_gen_07(const int num_frames, const int num_samples_per_pixel) {
    const double l = 25*SCHWARZSCHILD_RADIUS;

    const double latitude = 0;
    const double longitude0 = 0, longitude1 = pi;
    CameraSettings cs;
    cs.create_disk = false;
    cs.look_to = Point(ACCRETION_DISK_INNER_RADIUS*2, 0, 0);
    cs.num_samples_per_pixel = num_samples_per_pixel;
    cs.view_angle = 2*atan(SCHWARZSCHILD_RADIUS*7 / 2 / l)*2;
    vector<CameraSettings> all_cs;
    for (int i = 0; i < num_frames; ++i) {
        const double longitiude = linear_interpolation(i, 0, num_frames, longitude0, longitude1);
        cs.camera_position = Point(
            l * cos(latitude) * sin(longitiude),
            l * sin(latitude),
            l * cos(latitude) * cos(longitiude)
        );
        cs.time = linear_interpolation(i, 0, num_frames, 0, 1);
        cs.image_filepath = "../data/clip07_" + to_string(i) + ".png";
        all_cs.push_back(cs);
    }
    return all_cs;
}


void worker01() {
    // reduce image height from 800 to 100 for faster renders / testing
    Camera c(16./9, 800);
    const Image starmap("../data/starmap.png");
    const Disk3D diskmap = read_disk3d_from_file("../data/disk3d.txt");

    vector<CameraSettings> all_cs, _temp;
    // reduce num_frames and num_samples_per_pixel to 100 and 1 for faster renders / testing
    const int num_frames = 300, num_samples_per_pixel = 10;
    _temp = clip_gen_01(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());
    _temp = clip_gen_02(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());
    _temp = clip_gen_03(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());
    _temp = clip_gen_04(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());
    _temp = clip_gen_05(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());
    _temp = clip_gen_06(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());
    _temp = clip_gen_07(num_frames, num_samples_per_pixel); all_cs.insert(all_cs.end(), _temp.begin(), _temp.end());

    c.render(starmap, diskmap, all_cs);
}

int main() {
    set_black_hole_constants(1);
    // {
    //     const Point C(0.7, 0.7, 1);
    //     INNER_COLOR = C*3;
    //     OUTER_COLOR = C;
    // }
    {
        INNER_COLOR = Point(1, 0.8, 0.5)*3;
        OUTER_COLOR = Point(0.3, 0.1, 0);
    }
    BACKGROUND_COLOR = Point(0, 0, 0);
    worker01();
    return 0;
}
