//
// Created by viciopoli on 09.10.22.
//

#include "x/camera_models/camera_model.h"

namespace x {
    CameraModel::CameraModel(const Camera::Params &cameraParams) : cameraParams_(cameraParams) {}

    void CameraModel::undistort(FeatureList &features) {
        for (auto &f: features) {
            undistort(f);
        }
    }

    Matrix CameraModel::getCameraMatrix() const { return cameraParams_.getCameraMatrixEigen(); }

    cv::Matx33d CameraModel::getCVCameraMatrix() const { return cameraParams_.getCameraMatrix(); }

#ifdef MULTI_UAV
    cv::KeyPoint CameraModel::normalize(const cv::KeyPoint &point) const {
      return {static_cast<float>(point.pt.x * cameraParams_.inv_fx_ - cameraParams_.cx_n_),
              static_cast<float>(point.pt.y * cameraParams_.inv_fy_ - cameraParams_.cy_n_), point.size};
    }

    cv::KeyPoint CameraModel::denormalize(const cv::KeyPoint &point) const {
      return {static_cast<float>((point.pt.x + cameraParams_.cx_n_) / cameraParams_.inv_fx_),
              static_cast<float>((point.pt.y + cameraParams_.cy_n_) / cameraParams_.inv_fy_), point.size};
    }
#endif


    Feature CameraModel::normalize(const Feature &feature) const {
        Feature normalized_feature(feature.getTimestamp(),
                                   feature.getX() * cameraParams_.inv_fx_ - cameraParams_.cx_n_,
                                   feature.getY() * cameraParams_.inv_fy_ - cameraParams_.cy_n_, 0.0);
        normalized_feature.setXDist(feature.getXDist() * cameraParams_.inv_fx_ - cameraParams_.cx_n_);
        normalized_feature.setYDist(feature.getYDist() * cameraParams_.inv_fy_ - cameraParams_.cx_n_);

#ifdef MULTI_UAV
        cv::Mat d = feature.getDescriptor();
        assert(!d.empty());
        normalized_feature.setDescriptor(d);
#endif

#ifdef GT_DEBUG
        Vector3 landmark = feature.getLandmark();
        normalized_feature.setLandmark(landmark);
#endif

        return normalized_feature;
    }

    Track CameraModel::normalize(const Track &track, const size_t max_size) const {
        // Determine output track size
        const size_t track_size = track.size();
        size_t size_out;
        if (max_size) {
            size_out = std::min(max_size, track_size);
        } else {
            size_out = track_size;
        }
        Track normalized_track(size_out, Feature(), track.getId());
        const size_t start_idx(track_size - size_out);
        for (size_t j = start_idx; j < track_size; ++j) {
            normalized_track[j - start_idx] = normalize(track[j]);
        }
        return normalized_track;
    }

    TrackList CameraModel::normalize(const TrackList &tracks, const size_t max_size) const {
        const size_t n = tracks.size();
        TrackList normalized_tracks(n, Track());

        for (size_t i = 0; i < n; ++i) {
            normalized_tracks[i] = normalize(tracks[i], max_size);
        }

        return normalized_tracks;
    }

} // x