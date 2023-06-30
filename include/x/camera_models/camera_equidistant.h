//
// Created by viciopoli on 09.10.22.
//

#ifndef X_CAMERA_EQUIDISTANT_H
#define X_CAMERA_EQUIDISTANT_H

#include "camera_model.h"

namespace x {
    class CameraEquidistant : public CameraModel {
    public:
        explicit CameraEquidistant(const Camera::Params &params) : CameraModel(params) {}

        ~CameraEquidistant() override = default;

        void undistort(Feature &feature) override;

    private:
    };
} // x


#endif //X_CAMERA_EQUIDISTANT_H
