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

#ifndef X_FEATURE_H
#define X_FEATURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>

#ifdef GT_DEBUG
#include <Eigen/Dense>
#include <memory>
#endif

namespace x {
class Feature {
 public:
  /************************** Constructors **************************/

  Feature();

  Feature(const double &timestamp, double x, double y, double intensity);

  Feature(const double &timestamp, unsigned int frame_number, double x,
          double y, double x_dist, double y_dist, double intensity);

  Feature(const double &timestamp, unsigned int frame_number, double x_dist,
          double y_dist, unsigned int pyramid_level, float fast_score,
          double intensity);

  /******************** Overload Comparison operator ****************/

  bool operator==(const Feature &other);

  /**************************** Setters *****************************/

  void setX(const double x) { x_ = x; };

  void setY(const double y) { y_ = y; };

  void setXDist(const double x_dist) { x_dist_ = x_dist; };

  void setYDist(const double y_dist) { y_dist_ = y_dist; };

  void setTile(int row, int col) {
    tile_row_ = row;
    tile_col_ = col;
  };

#ifdef GT_DEBUG
  void setLandmark(const Eigen::Vector3d landmark) { landmark_ = landmark; }
#endif

#ifdef MULTI_UAV
  void setDescriptor(const cv::Mat &descriptor) { descriptor_ = descriptor; };
#endif

  /**************************** Getters *****************************/

  [[nodiscard]] double getTimestamp() const { return timestamp_; };

  [[nodiscard]] double getX() const { return x_; };

  [[nodiscard]] double getY() const { return y_; }

  [[nodiscard]] double getXDist() const { return x_dist_; };

  [[nodiscard]] double getYDist() const { return y_dist_; };

  [[nodiscard]] cv::Point2d getDistPoint2d() const {
    return {x_dist_, y_dist_};
  };

  [[nodiscard]] cv::Point2d getDistPoint2f() const {
    return cv::Point2f(static_cast<float>(x_dist_),
                       static_cast<float>(y_dist_));
  };

  [[nodiscard]] cv::Point2d getPoint2d() const { return {x_, y_}; };

  [[nodiscard]] unsigned int getPyramidLevel() const { return pyramid_level_; };

  [[nodiscard]] float getFastScore() const { return fast_score_; };

  [[nodiscard]] int getTileRow() const { return tile_row_; };

  [[nodiscard]] int getTileCol() const { return tile_col_; };

    // #ifdef GT_DEBUG
    //   std::shared_ptr<Feature> getMatch() const { return other_match_; }
    // #endif
#ifdef GT_DEBUG
  Eigen::Vector3d getLandmark() const { return landmark_; }
#endif

#ifdef MULTI_UAV
  [[nodiscard]] cv::Mat getDescriptor() const { return descriptor_; };
#endif

  [[nodiscard]] double getIntensity() const { return intensity_; };

 private:
  bool nearlyEqual(double a, double b);

  /**
   * Image timestamp [s]
   */
  double timestamp_{0};

  /**
   * Image ID
   */
  unsigned int frame_number_ = 0;

  /**
   * (Undistorted) X pixel coordinate
   */
  double x_ = 0.0;

  /**
   * (Undistorted) Y pixel coordinate
   */
  double y_ = 0.0;

  /**
   * Distorted X pixel coordinate
   */
  double x_dist_ = 0.0;

  /**
   * Distorted Y pixel coordinate
   */
  double y_dist_ = 0.0;

  /**
   * Pyramid level in which the feature was detected
   */
  unsigned int pyramid_level_ = 0;

  /**
   * FAST score
   */
  float fast_score_ = 0.0;

  /**
   * Row index of the image tile in which the feature is located
   */
  int tile_row_ = -1;

  /**
   * Column index of the image tile in which the feature is located
   */
  int tile_col_ = -1;

#ifdef GT_DEBUG
  Eigen::Vector3d landmark_ = Eigen::Vector3d(-1.0, -1.0, -1.0);
#endif

#ifdef MULTI_UAV
  /**
   * Feature descriptor for place recognition
   */
  cv::Mat descriptor_ =
      cv::Mat::zeros(1, 32, CV_8UC1);  // 32 is the length of ORB descriptor
#endif

  /**
   * Pixel intensities for photometric calibration
   */
  double intensity_ = -1.0;
};
}  // namespace x

#endif