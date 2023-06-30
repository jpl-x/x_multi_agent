//
// Created by viciopoli on 09.10.22.
//

#include "x/camera_models/camera_equidistant.h"

using namespace x;

void CameraEquidistant::undistort(Feature &feature) {

    const std::vector<cv::Point2d> src{feature.getDistPoint2f()};
    std::vector<cv::Point2d> res;
    cv::fisheye::undistortPoints(src, res, cameraParams_.getCameraMatrix(), cameraParams_.dist_coeff);

    feature.setX(res[0].x * cameraParams_.fx_ + cameraParams_.cx_);
    feature.setY(res[0].y * cameraParams_.fy_ + cameraParams_.cy_);
}


