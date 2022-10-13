/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "x/vision/camera.h"

#include <Eigen/Core>
#include <iostream>
#include <opencv2/core/eigen.hpp>

using namespace x;

Camera::Camera() = default;

Camera::Camera(double fx, double fy, double cx, double cy, double s,
               unsigned int img_width, unsigned int img_height)
    : s_(s), img_width_(img_width), img_height_(img_height) {
  fx_ = img_width * fx;
  fy_ = img_height * fy;
  cx_ = img_width * cx;
  cy_ = img_height * cy;

  inv_fx_ = 1.0 / fx_;
  inv_fy_ = 1.0 / fy_;
  cx_n_ = cx_ * inv_fx_;
  cy_n_ = cy_ * inv_fy_;
  s_term_ = 1.0 / (2.0 * std::tan(s / 2.0));

  K_ = Matrix::Zero(3, 3);
  K_(0, 0) = fx_;
  K_(1, 1) = fy_;
  K_(0, 2) = cx_;
  K_(1, 2) = cy_;
  K_(2, 2) = 1.0;
  cv::eigen2cv(K_, cvK_);
}

unsigned int Camera::getWidth() const { return img_width_; }

unsigned int Camera::getHeight() const { return img_height_; }

double Camera::getInvFx() const { return inv_fx_; }

double Camera::getInvFy() const { return inv_fy_; }

double Camera::getCxN() const { return cx_n_; }

double Camera::getCyN() const { return cy_n_; }

void Camera::undistort(FeatureList &features) const {
  // Undistort each point in the input vector
  for (auto &feature : features) {
    undistort(feature);
  }
}

void Camera::undistort(Feature &feature) const {
  // cv::Point2d p =
  //     LUT_calib_[feature.getYDist() * img_width_ + feature.getXDist()];
  // feature.setX(p.x);
  // feature.setY(p.y);
  const double cam_dist_x = feature.getXDist() * inv_fx_ - cx_n_;
  const double cam_dist_y = feature.getYDist() * inv_fy_ - cy_n_;

  const double dist_r = sqrt(cam_dist_x * cam_dist_x + cam_dist_y * cam_dist_y);

  double distortion_factor = 1.0;
  if (dist_r > 0.01) distortion_factor = inverseTf(dist_r) / dist_r;

  const double xn = distortion_factor * cam_dist_x;
  const double yn = distortion_factor * cam_dist_y;

  feature.setX(xn * fx_ + cx_);
  feature.setY(yn * fy_ + cy_);
}

Matrix Camera::getCameraMatrix() const { return K_; }
cv::Mat Camera::getCVCameraMatrix() const { return cvK_; }
#ifdef MULTI_UAV

cv::KeyPoint Camera::normalize(const cv::KeyPoint &point) const {
  return {static_cast<float>(point.pt.x * inv_fx_ - cx_n_),
          static_cast<float>(point.pt.y * inv_fy_ - cy_n_), point.size};
}
cv::KeyPoint Camera::denormalize(const cv::KeyPoint &point) const {
  return {static_cast<float>((point.pt.x + cx_n_) / inv_fx_),
          static_cast<float>((point.pt.y + cy_n_) / inv_fy_), point.size};
}
#endif

Feature Camera::normalize(const Feature &feature) const {
  Feature normalized_feature(
      feature.getTimestamp(), feature.getX() * inv_fx_ - cx_n_,
      feature.getY() * inv_fy_ - cy_n_, feature.getIntensity());

#ifdef MULTI_UAV
  cv::Mat d = feature.getDescriptor();
  assert(!d.empty());
  normalized_feature.setDescriptor(d);
#endif

#ifdef GT_DEBUG
  Vector3 landmark = feature.getLandmark();
  normalized_feature.setLandmark(landmark);
#endif

  normalized_feature.setXDist(feature.getXDist() * inv_fx_ - cx_n_);
  normalized_feature.setYDist(feature.getYDist() * inv_fy_ - cy_n_);

  return normalized_feature;
}

/** \brief Normalized the image coordinates for all features in the input track
 *  \param track Track to normalized
 *  \param max_size Max size of output track, cropping from the end (default 0:
 * no cropping) \return Normalized track
 */
Track Camera::normalize(const Track &track, const size_t max_size) const {
  // Determine output track size
  const size_t track_size = track.size();
  size_t size_out;
  if (max_size)
    size_out = std::min(max_size, track_size);
  else
    size_out = track_size;

  Track normalized_track(size_out, Feature(), track.getId());
  const size_t start_idx(track_size - size_out);
  for (size_t j = start_idx; j < track_size; ++j) {
    normalized_track[j - start_idx] = normalize(track[j]);
  }
  return normalized_track;
}

/** \brief Normalized the image coordinates for all features in the input track
 * list \param tracks List of tracks \param max_size Max size of output tracks,
 * cropping from the end (default 0: no cropping) \return Normalized list of
 * tracks
 */
TrackList Camera::normalize(const TrackList &tracks,
                            const size_t max_size) const {
  const size_t n = tracks.size();
  TrackList normalized_tracks;

  for (size_t i = 0; i < n; ++i)
    normalized_tracks.push_back(normalize(tracks[i], max_size));

  return normalized_tracks;
}

double Camera::inverseTf(const double &dist) const {
  if (s_ == 0.0) {
    return dist;
  } else {
    return std::tan(dist * s_) * s_term_;
  }
}
