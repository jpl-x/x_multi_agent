//
// Created by viciopoli on 09.10.22.
//

#include "x/camera_models/camera_radtan.h"

using namespace x;

void CameraRadTan::undistort(Feature &feature) {
    const std::vector<cv::Point2d> src{feature.getDistPoint2f()};

    std::vector<cv::Point2d> res;
    const cv::Matx33d m = cameraParams_.getCameraMatrix();

    cv::undistortPoints(src, res, m, cameraParams_.dist_coeff);

    feature.setX(res[0].x * cameraParams_.fx_ + cameraParams_.cx_);
    feature.setY(res[0].y * cameraParams_.fy_ + cameraParams_.cy_);
}
