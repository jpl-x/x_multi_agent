//
// Created by viciopoli on 09.10.22.
//

#ifndef X_CAMERA_NONE_H
#define X_CAMERA_NONE_H

#include "camera_model.h"

namespace x {
    class CameraNone : public CameraModel {
    public:
        explicit CameraNone(const Camera::Params &params) : CameraModel(params) {}

        ~CameraNone() override = default;

        void undistort(Feature &feature) override;

    private:
    };
} // x

#endif //X_CAMERA_NONE_H
