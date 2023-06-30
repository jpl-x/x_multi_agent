//
// Created by viciopoli on 09.10.22.
//

#include "x/camera_models/camera_fov.h"

#include <cmath>

using namespace x;

void CameraFov::undistort(Feature &feature) {
    const double cam_dist_x = feature.getXDist() * cameraParams_.inv_fx_ - cameraParams_.cx_n_;
    const double cam_dist_y = feature.getYDist() * cameraParams_.inv_fy_ - cameraParams_.cy_n_;

    const double dist_r = sqrt(cam_dist_x * cam_dist_x + cam_dist_y * cam_dist_y);

    double distortion_factor = 1.0;
    if (dist_r > 0.01) {
        distortion_factor = inverseTf(dist_r) / dist_r;
    }
    const double xn = distortion_factor * cam_dist_x;
    const double yn = distortion_factor * cam_dist_y;

    feature.setX(xn * cameraParams_.fx_ + cameraParams_.cx_);
    feature.setY(yn * cameraParams_.fy_ + cameraParams_.cy_);
}


double CameraFov::inverseTf(const double dist) const {
    if (cameraParams_.dist_coeff[0] != 0.0) {
        return std::tan(dist * cameraParams_.dist_coeff[0]) * s_term_;
    } else {
        return dist;
    }
}
