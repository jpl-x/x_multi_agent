//
// Created by viciopoli on 09.10.22.
//

#ifndef X_CAMERA_RAD_TAN_H
#define X_CAMERA_RAD_TAN_H

#include "camera_model.h"

namespace x {
    class CameraRadTan : public CameraModel {
    public:
        explicit CameraRadTan(const Camera::Params &params) : CameraModel(params) {}

        ~CameraRadTan() override = default;

        void undistort(Feature &feature) override;

    private:
    };
} // x

#endif //X_CAMERA_RAD_TAN_H
