//
// Created by viciopoli on 09.10.22.
//

#ifndef X_CAMERA_FOV_H
#define X_CAMERA_FOV_H

#include "camera_model.h"

namespace x {
    class CameraFov : public CameraModel {
    public:
        CameraFov(const Camera::Params &params) : CameraModel(params) {
            s_term_ = 1.0 / (2.0 * std::tan(params.dist_coeff[0] / 2.0));
        }

        ~CameraFov() override = default;

        void undistort(Feature &feature) override;

    private:
        double s_term_;

        // Inverse FOV distortion transformation
        [[nodiscard]] double inverseTf(double dist) const;
    };
} // x

#endif //X_CAMERA_FOV_H
