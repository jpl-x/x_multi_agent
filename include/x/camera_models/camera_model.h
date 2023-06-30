//
// Created by viciopoli on 09.10.22.
//

#ifndef X_CAMERAMODEL_H
#define X_CAMERAMODEL_H

#include "x/vision/feature.h"
#include "x/vision/types.h"
#include <utility>
#include <vector>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "x/camera_models/types.h"

namespace x {
    /**
     * Base class
     */
    class CameraModel {
    public:

        explicit CameraModel(const Camera::Params &params);

        virtual ~CameraModel() = default;

        void set(const Camera::Params &cameraParams) {
            cameraParams_ = cameraParams;
        };

        void undistort(FeatureList &features);

        virtual void undistort(Feature &feature) = 0;

        [[nodiscard]] unsigned int getWidth() const { return cameraParams_.img_width_; };

        [[nodiscard]] unsigned int getHeight() const { return cameraParams_.img_height_; };

        [[nodiscard]] double getFx() const { return cameraParams_.fx_; };

        [[nodiscard]] double getFy() const { return cameraParams_.fy_; };

        [[nodiscard]] Feature normalize(const Feature &feature) const;

        [[nodiscard]] Track normalize(const Track &track, size_t max_size = 0) const;

        [[nodiscard]] TrackList normalize(const TrackList &tracks, size_t max_size = 0) const;

        [[nodiscard]] Matrix getCameraMatrix() const;

        [[nodiscard]] cv::Matx33d getCVCameraMatrix() const;

#ifdef MULTI_UAV
        [[nodiscard]] cv::KeyPoint normalize(const cv::KeyPoint &point) const;
        [[nodiscard]] cv::KeyPoint denormalize(const cv::KeyPoint &point) const;
#endif

    protected:
        Camera::Params cameraParams_;

    private:

    };
} // x

#endif //X_CAMERAMODEL_H
