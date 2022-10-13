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

#ifndef JPL_VPP_TRACKER_H_
#define JPL_VPP_TRACKER_H_

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

#include "x/vision/camera.h"
#include "x/vision/types.h"

#ifdef MULTI_UAV
#include "x/place_recognition/place_recognition.h"
#endif

#ifdef PHOTOMETRIC_CALI
#include <boost/thread.hpp>

#include "x/photometric_calibration/irPhotoCalib.h"
#endif

namespace x {
class Tracker {
 public:
  ~Tracker();

  Tracker();

  /**
   * Initiaize Tracker and its parameters
   * @param cam
   * @param fast_detection_delta
   * @param non_max_supp
   * @param block_half_length
   * @param margin
   * @param n_feat_min
   * @param outlier_method
   * @param outlier_param1
   * @param outlier_param2
   * @param win_size_w
   * @param win_size_h
   * @param max_level
   * @param min_eig_thr
   * @param place_recognition
   * @param temporal_params_div
   * @param spatial_params
   * @param spatial_params_thr
   * @param epsilon_gap
   * @param epsilon_base
   * @param max_level_photo
   * @param min_eig_thr_photo
   * @param win_size_w_photo
   * @param win_size_h_photo
   * @param fast_detection_delta_photo
   */
  Tracker(const Camera &cam, int fast_detection_delta, bool non_max_supp,
          unsigned int block_half_length, unsigned int margin,
          unsigned int n_feat_min, int outlier_method, double outlier_param1,
          double outlier_param2, int win_size_w, int win_size_h, int max_level,
          double min_eig_thr
#ifdef MULTI_UAV
          ,
          std::shared_ptr<PlaceRecognition> &place_recognition
#endif
#ifdef PHOTOMETRIC_CALI
          ,
          int temporal_params_div = 0, bool spatial_params = false,
          double spatial_params_thr = 0.0, double epsilon_gap = 0.5,
          double epsilon_base = 0.4, int max_level_photo = 2,
          double min_eig_thr_photo = 0.003, int win_size_w_photo = 31,
          int win_size_h_photo = 31, int fast_detection_delta_photo = 30
#endif
  );

  /**
   * Set tracker parameters
   *
   * @param cam
   * @param fast_detection_delta
   * @param non_max_supp
   * @param block_half_length
   * @param margin
   * @param n_feat_min
   * @param outlier_method
   * @param outlier_param1
   * @param outlier_param2
   * @param win_size_w
   * @param win_size_h
   * @param max_level
   * @param min_eig_thr
   * @param temporal_params_div
   * @param spatial_params
   * @param spatial_params_thr
   * @param epsilon_gap
   * @param epsilon_base
   * @param max_level_photo
   * @param min_eig_thr_photo
   * @param win_size_w_photo
   * @param win_size_h_photo
   * @param fast_detection_delta_photo
   */
  void setParams(const Camera &cam, int fast_detection_delta, bool non_max_supp,
                 unsigned int block_half_length, unsigned int margin,
                 unsigned int n_feat_min, int outlier_method,
                 double outlier_param1, double outlier_param2, int win_size_w,
                 int win_size_h, int max_level, double min_eig_thr
#ifdef PHOTOMETRIC_CALI
                 ,
                 int temporal_params_div = 0, bool spatial_params = false,
                 double spatial_params_thr = 0.0, double epsilon_gap = 0.5,
                 double epsilon_base = 0.4, int max_level_photo = 2,
                 double min_eig_thr_photo = 0.003, int win_size_w_photo = 31,
                 int win_size_h_photo = 31, int fast_detection_delta_photo = 30
#endif
  );

  /**
   * Get latest matches
   *
   * @return MatchList
   */
  [[nodiscard]] MatchList getMatches() const;

  /**
   * Get ID of last image processed
   *
   * @return std::optional<int>
   */
  [[nodiscard]] std::optional<int> getImgId() const;

  /**
   * Tracks features from last image into the current one
   *
   * @param current_img
   * @param timestamp
   * @param frame_number
   */
  void track(TiledImage &current_img, const double &timestamp,
             unsigned int frame_number);

  /**
   * Check if new matches are available
   *
   * @return bool
   */
  bool checkMatches();

  /**
   * Plots matches in the image
   *
   * @param matches
   * @param img
   */
  static void plotMatches(MatchList matches, TiledImage &img);

#ifdef PHOTOMETRIC_CALI
  /**
   * Apply the photometric calibration to the thermal image
   *
   * @param tracks
   */
  void setIntensistyHistory(const TrackList &tracks);
#endif

#ifdef MULTI_UAV
  /**
   * Add keyframe to the database
   *
   * @param keyframe
   */
  void addKeyframe(const KeyframePtr &keyframe);

  /**
   * Clean SLAM matches
   */
  void cleanSlamMatches();

  /**
   * Get MSCK matches reference
   *
   * @return MsckfMatches
   */
  MsckfMatches &getMsckfMatches();

  /**
   * Get SLAM matches reference
   *
   * @return SlamMatches
   */
  SlamMatches &getSlamMatches();

  /**
   * Upgrade to MSCKF/SLAM or delete opportunistic matches
   *
   * @param current_msckf_tracks
   * @param current_slam_tracks
   * @param current_opp_tracks
   */
  void updateOppMatches(const TrackList &current_msckf_tracks,
                        const TrackList &current_slam_tracks,
                        const TrackList &current_opp_tracks);

  /**
   * Get opportunistic tracks ids
   *
   * @return
   */
  OppIDListPtr getOppIds();
#endif

 private:
  // int img_id_store_frame_=0;

  // FAST features detector settings
  cv::Size win_size_ = cv::Size(31, 31);
  cv::TermCriteria term_crit_ = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

  MatchList matches_;
  int img_id_ = 0;
  FeatureList previous_features_;
  double previous_timestamp_{0};
  TiledImage previous_img_;
  // Tracker params
  Camera camera_;
  int fast_detection_delta_ =
      9;  // the intensity difference threshold for the FAST feature detector
  bool non_max_supp_ = true;  // if true, non-maximum suppression is applied to
  // detected corners (keypoints)
  unsigned int block_half_length_ =
      20;  // the blocked region for other features to occupy will be of size (2
  // * block_half_length_ + 1)^2 [px]
  unsigned int margin_ = 20;  // the margin from the edge of the image within
  // which all detected features must fall into
  unsigned int n_feat_min_ =
      40;  // min number of feature, triggers the feature research
  int outlier_method_ = 8;
  double outlier_param1_ = 0.3;
  double outlier_param2_ = 0.99;

  int max_level_ = 2;           // max level KLT pyramid
  double min_eig_thr_ = 0.003;  // minEigThreshold KLT

#ifdef MULTI_UAV
  // place recognition params
  std::shared_ptr<PlaceRecognition> place_recognition_;
#endif

  // Placeholder members from an older version to enable multi-scale pyramid
  // detection and tracking
  // TODO(Jeff or Roland): implement based on best Mars Heli test results
  ImagePyramid img_pyramid_;
  unsigned int pyramid_depth_ = 1;  // depth of the half-sampling pyramid
  // Feature detection. One can avoid detection in the neighborhood of features
  // already tracked and specified in the optional argument old_points.
  void featureDetection(const TiledImage &img, FeatureList &new_pts,
                        const double &timestamp, unsigned int frame_number,
                        FeatureList &old_pts);

  // Get the image pyramid given the highest resolution image using
  // down-sampling
  void getImagePyramid(const TiledImage &img, ImagePyramid &pyramid) const;

  // Get FAST features for an image pyramid at all levels. Optionally avoids
  // detection on the neighborhood of old features.
  void getFASTFeaturesPyramid(ImagePyramid &pyramid, const double &timestamp,
                              unsigned int frame_number, FeatureList &features,
                              FeatureList &old_features);

  // Get fast features for a single image pyramid level. Avoids detection
  // on the neighborhood of old features, if applicable.
  void getFASTFeaturesImage(TiledImage &img, const double &timestamp,
                            unsigned int frame_number,
                            unsigned int pyramid_level, FeatureList &features,
                            FeatureList &old_features);

  // Computes the neighborhood mask for the feature list at input.
  void computeNeighborhoodMask(const FeatureList &features, const cv::Mat &img,
                               cv::Mat &mask) const;

  // Check whether a feature falls within a certain border at a certain pyramid
  // level
  bool isFeatureInsideBorder(Feature &feature, const TiledImage &img,
                             unsigned int margin,
                             unsigned int pyramid_level) const;

  // Get scaled image coordinate for a feature
  static void getScaledPixelCoordinate(const Feature &feature,
                                       PixelCoor &scaled_coord);

  // Appends candidate features which are far enough from existing features. The
  // neighborhood mask needs to be provided beforehands. The order of the
  // features matters.
  void appendNonNeighborFeatures(TiledImage &img, FeatureList &features,
                                 FeatureList &candidate_features,
                                 cv::Mat &mask) const;

  void removeOverflowFeatures(TiledImage &img1, TiledImage &img2,
                              FeatureList &features1,
                              FeatureList &features2) const;

  // Feature tracking using KLT
  void featureTracking(const cv::Mat &img1, const cv::Mat &img2,
                       const cv::Mat &img2_origin, FeatureList &pts1,
                       FeatureList &pts2, const double &timestamp2,
                       unsigned int frame_number2) const;

  TiledImage current_img_origin_;

#ifdef PHOTOMETRIC_CALI

  std::vector<std::vector<float>> tracks_intensity_history_,
      tracks_intensity_current_;
  std::vector<std::vector<std::pair<int, int>>> points_prev_, points_curr_;
  std::vector<int> frame_diff_history_;

  bool CALIBRATION_DONE = false;
  int update_counter = 0;
  // window size to compute the intensity value changes to track over time
  int temporal_params_div_ = 16;
  // true->compute the spatial parameters
  bool spatial_params_ = true;
  // the features must cover the spatial_params_thr
  // percentage of the image to compute the params
  double spatial_params_thr_ = 0.9;

  double epsilon_gap_ = 0.5;
  double epsilon_base_ = 0.4;

  // KLT tracker params for calibrated images
  int max_level_photo_ = 2;
  double min_eig_thr_photo_ = 0.003;
  cv::Size win_size_photo_ = cv::Size(31, 31);
  int fast_detection_delta_photo_ = 30;

  std::vector<std::vector<std::pair<int, int>>> prev_pixels_history_,
      current_pixels_current_;
  std::vector<std::pair<int, int>> pair_prev_, pair_curr_;
  std::vector<std::vector<float>> prev_intensity_history_,
      current_intensity_history_;

  std::vector<float> intensities_prev_, intensities_curr_;

  // photometric calibration parameters
  // PhotoParams prev_photo_params_;
  std::shared_ptr<IRPhotoCalib> calibrator_;
  int intensities_kernel_size_ = 30;

  double k_epsilon_gap_ = 0.01;
  double k_epsilon_base_ = 0.01;

  void calibrateImage(TiledImage &img1, TiledImage &img2);

  [[nodiscard]] float computeIntensity(const cv::Mat &img, int x, int y) const;

  TiledImage prev_img_origin_;

  void refinePhotometricParams(const TrackList &tracks);
#endif
};
}  // namespace x
#endif
