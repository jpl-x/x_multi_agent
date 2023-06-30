//
// Created by viciopoli on 05.01.23.
//

#ifndef X_CAMERA_H
#define X_CAMERA_H

#include "x/camera_models/camera_model.h"
#include "x/camera_models/camera_equidistant.h"
#include "x/camera_models/camera_radtan.h"
#include "x/camera_models/camera_fov.h"
#include "x/camera_models/camera_none.h"

namespace x::Camera {
    static std::shared_ptr<CameraModel> constructCamera(const Params &params) {
        // Initialize camera geometry
        if (params.cameraType == Camera::DistortionModel::RADTAN) {
            return std::make_shared<CameraRadTan>(params);
        } else if (params.cameraType == Camera::DistortionModel::EQUIDISTANT) {
            return std::make_shared<CameraEquidistant>(params);
        } else if (params.cameraType == Camera::DistortionModel::FOV) {
            return std::make_shared<CameraFov>(params);
        } else {
            return std::make_shared<CameraNone>(params);
        }
    }
}

#endif //X_CAMERA_H
