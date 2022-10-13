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

#include "x/vision/tracker.h"

#include <boost/log/trivial.hpp>
#include <cmath>
#include <memory>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>

using namespace x;

Tracker::~Tracker() {}

Tracker::Tracker() = default;

Tracker::Tracker(const Camera &cam, const int fast_detection_delta,
                 const bool non_max_supp, const unsigned int block_half_length,
                 unsigned int margin, unsigned int n_feat_min,
                 int outlier_method, double outlier_param1,
                 double outlier_param2, int win_size_w, int win_size_h,
                 int max_level, double min_eig_thr
#ifdef MULTI_UAV
                 ,
                 std::shared_ptr<PlaceRecognition> &place_recognition
#endif
#ifdef PHOTOMETRIC_CALI
                 ,
                 const int temporal_params_div, const bool spatial_params,
                 const double spatial_params_thr, const double epsilon_gap,
                 const double epsilon_base, const int max_level_photo,
                 const double min_eig_thr_photo, const int win_size_w_photo,
                 const int win_size_h_photo,
                 const int fast_detection_delta_photo
#endif
                 )
    : win_size_(cv::Size(win_size_w, win_size_h)),
      camera_(cam),
      fast_detection_delta_(fast_detection_delta),
      non_max_supp_(non_max_supp),
      block_half_length_(block_half_length),
      margin_(margin),
      n_feat_min_(n_feat_min),
      outlier_method_(outlier_method),
      outlier_param1_(outlier_param1),
      outlier_param2_(outlier_param2),
      max_level_(max_level),
      min_eig_thr_(min_eig_thr)
#ifdef MULTI_UAV
      ,
      place_recognition_(place_recognition)
#endif
#ifdef PHOTOMETRIC_CALI
      ,
      temporal_params_div_(temporal_params_div),
      spatial_params_(spatial_params),
      spatial_params_thr_(spatial_params_thr),
      epsilon_gap_(epsilon_gap),
      epsilon_base_(epsilon_base),
      max_level_photo_(max_level_photo),
      min_eig_thr_photo_(min_eig_thr_photo),
      win_size_photo_(cv::Size(win_size_w_photo, win_size_h_photo)),
      fast_detection_delta_photo_(fast_detection_delta_photo)
#endif
{
#ifdef PHOTOMETRIC_CALI
  calibrator_ = std::make_shared<IRPhotoCalib>(
      camera_.getWidth(), camera_.getHeight(), temporal_params_div_,
      spatial_params_, spatial_params_thr_, epsilon_gap_, epsilon_base_, false);
#endif
  img_pyramid_.reserve(pyramid_depth_);  // prealloc memory for pyramid vector
}

void Tracker::setParams(
    const Camera &cam, const int fast_detection_delta, const bool non_max_supp,
    const unsigned int block_half_length, const unsigned int margin,
    const unsigned int n_feat_min, const int outlier_method,
    const double outlier_param1, const double outlier_param2,
    const int win_size_w, const int win_size_h, const int max_level,
    const double min_eig_thr
#ifdef PHOTOMETRIC_CALI
    ,
    int temporal_params_div, bool spatial_params, double spatial_params_thr,
    double epsilon_gap, double epsilon_base, int max_level_photo,
    double min_eig_thr_photo, int win_size_w_photo, int win_size_h_photo,
    int fast_detection_delta_photo
#endif
) {
  camera_ = cam;
  fast_detection_delta_ = fast_detection_delta;
  non_max_supp_ = non_max_supp;
  block_half_length_ = block_half_length;
  margin_ = margin;
  n_feat_min_ = n_feat_min;
  outlier_method_ = outlier_method;
  outlier_param1_ = outlier_param1;
  outlier_param2_ = outlier_param2;
  win_size_ = cv::Size(win_size_w, win_size_h);
  max_level_ = max_level;
  min_eig_thr_ = min_eig_thr;

#ifdef PHOTOMETRIC_CALI
  temporal_params_div_ = temporal_params_div;
  spatial_params_ = spatial_params;
  spatial_params_thr_ = spatial_params_thr;
  epsilon_gap_ = epsilon_gap;
  epsilon_base_ = epsilon_base;
  max_level_photo_ = max_level_photo;
  min_eig_thr_photo_ = min_eig_thr_photo;
  win_size_photo_ = cv::Size(win_size_w_photo, win_size_h_photo);
  fast_detection_delta_photo_ = fast_detection_delta_photo;
#endif
}

MatchList Tracker::getMatches() const { return matches_; }

std::optional<int> Tracker::getImgId() const { return img_id_; }

void Tracker::track(TiledImage &current_img, const double &timestamp,
                    unsigned int frame_number) {
  // Increment the current image number
  img_id_++;

#ifdef TIMING
  // Set up and start internal timing
  clock_t clock1, clock2, clock3, clock4, clock5, clock6, clock7, clock8;
  clock1 = clock();
#endif

  // Build image pyramid
  getImagePyramid(current_img, img_pyramid_);

#ifdef TIMING
  clock2 = clock();
#endif

  //============================================================================
  // Feature detection and tracking
  //============================================================================

  FeatureList current_features;
  MatchList matches;

  // If first image
  if (img_id_ == 1) {
#ifdef PHOTOMETRIC_CALI
    current_img_origin_ = current_img;
#endif
    // Detect features in first image
    FeatureList old_features;
    featureDetection(current_img, current_features, timestamp, frame_number,
                     old_features);
    // Undistortion of current features
    camera_.undistort(current_features);
#ifdef TIMING
    clock3 = clock();

    // Deal with unused timers
    clock4 = clock3;
    clock5 = clock4;
    clock6 = clock5;
#endif
  } else  // not first image
  {
#ifdef TIMING
    // End detection timer even if it did not occur
    clock3 = clock();
#endif

#ifdef PHOTOMETRIC_CALI
    current_img.copyTo(current_img_origin_);

    calibrateImage(prev_img_origin_, current_img);  // calibrate the image

#endif

    // Track features
    featureTracking(previous_img_, current_img, current_img_origin_,
                    previous_features_, current_features, timestamp,
                    frame_number);
#ifdef TIMING
    clock4 = clock();
#endif

    removeOverflowFeatures(previous_img_, current_img, previous_features_,
                           current_features);

    // Refresh features if needed
    if (current_features.size() < n_feat_min_) {
#ifdef DEBUG
      BOOST_LOG_TRIVIAL(info) << "Number of tracked features reduced to "
                              << current_features.size() << std::endl;
      BOOST_LOG_TRIVIAL(info) << "Triggering re-detection" << std::endl;
#endif

      // Detect and track new features outside of the neighborhood of
      // previously-tracked features
      FeatureList previous_features_new, current_features_new;
      featureDetection(previous_img_, previous_features_new,
                       previous_img_.getTimestamp(),
                       previous_img_.getFrameNumber(), previous_features_);
      featureTracking(previous_img_, current_img, current_img_origin_,
                      previous_features_new, current_features_new, timestamp,
                      frame_number);

      // Concatenate previously-tracked and newly-tracked features
      previous_features_.insert(previous_features_.end(),
                                previous_features_new.begin(),
                                previous_features_new.end());
      current_features.insert(current_features.end(),
                              current_features_new.begin(),
                              current_features_new.end());
    }

#ifdef TIMING
    clock5 = clock();
#endif
    // Undistortion of current features
    camera_.undistort(current_features);
    // Undistort features in the previous image
    // TODO(jeff) only do it for new features newly detected
    camera_.undistort(previous_features_);

    //==========================================================================
    // Outlier removal
    //==========================================================================

    unsigned int n_matches = previous_features_.size();
    if (n_matches) {
      // TODO(jeff) Store coordinates as cv::Point2f in Feature class to avoid
      // theses conversions
      // Convert features to OpenCV keypoints
      std::vector<cv::Point2f> pts1, pts2;
      pts1.resize(n_matches);
      pts2.resize(n_matches);
      for (unsigned int i = 0; i < pts1.size(); i++) {
        pts1[i].x = static_cast<float>(previous_features_[i].getX());
        pts1[i].y = static_cast<float>(previous_features_[i].getY());
        pts2[i].x = static_cast<float>(current_features[i].getX());
        pts2[i].y = static_cast<float>(current_features[i].getY());
      }

      std::vector<uchar> mask;
      cv::findFundamentalMat(pts1, pts2, outlier_method_, outlier_param1_,
                             outlier_param2_, mask);

      FeatureList current_features_refined, previous_features_refined;
      for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != 0) {
          previous_features_refined.push_back(previous_features_[i]);
          current_features_refined.push_back(current_features[i]);
        }
      }

      std::swap(previous_features_refined, previous_features_);
      std::swap(current_features_refined, current_features);
    }
#ifdef TIMING
    clock6 = clock();
#endif

    //==========================================================================
    // Match construction
    //==========================================================================

    // Reset match list
    n_matches = previous_features_.size();
    matches.resize(n_matches);

    // Store the message array into a match list
    for (unsigned int i = 0; i < n_matches; ++i) {
      Match current_match;
      current_match.previous = previous_features_.at(i);
      current_match.current = current_features.at(i);

      // Add match to list
      matches[i] = current_match;
    }
  }

  // Store results so it can queried outside the class
  matches_ = matches;

  previous_features_ = current_features;
  previous_timestamp_ = timestamp;
  previous_img_ = current_img;

#ifdef PHOTOMETRIC_CALI
  current_img_origin_.copyTo(prev_img_origin_);
#endif

  // Exit the function if this is the first image
  if (img_id_ < 2) return;

#ifdef TIMING
  clock7 = clock();
#endif

  // Print TIMING_T info
#ifdef TIMING
  BOOST_LOG_TRIVIAL(info)
      << "Tracking Info ====================================" << std::endl;
  BOOST_LOG_TRIVIAL(info) << "Number of matches:                   "
                          << matches.size() << std::endl;
  BOOST_LOG_TRIVIAL(info)
      << "==================================================" << std::endl;
  BOOST_LOG_TRIVIAL(info) << "Times ===================================="
                          << std::endl
                          << "Build image pyramid:               "
                          << (double)(clock2 - clock1) / CLOCKS_PER_SEC * 1000
                          << " ms" << std::endl
                          << "Feature detection:                 "
                          << (double)(clock3 - clock2) / CLOCKS_PER_SEC * 1000
                          << " ms" << std::endl
#ifdef PHOTOMETRIC_CALI
                          << "Photometric calibration:                  "
                          << (double)(clock8 - clock3) / CLOCKS_PER_SEC * 1000
                          << " ms" << std::endl;
  clock3 = clock8;
#endif
  BOOST_LOG_TRIVIAL(info)
      << ""
      << "Feature tracking:                  "
      << (double)(clock4 - clock3) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl
      << "Feature re-detection and tracking: "
      << (double)(clock5 - clock4) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl
      << "Outlier check:                     "
      << (double)(clock6 - clock5) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl
      << "Feature conversion:                "
      << (double)(clock7 - clock6) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl
      << "------------------------------------------" << std::endl
      << "TOTAL:                             "
      << (double)(clock7 - clock1) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl
      << "==========================================" << std::endl;
#endif
}

/** \brief True if matches are chronological order
 *  @todo Make Match struct a class with this function in it.
 */
bool Tracker::checkMatches() {
  if (!matches_.empty()) {
    Match match(matches_[0]);
    if (match.previous.getTimestamp() < match.current.getTimestamp()) {
      return true;
    }
  }
  return false;
}

void Tracker::plotMatches(MatchList matches, TiledImage &img) {
  // Convert grayscale image to color image
#if CV_MAJOR_VERSION == 4
  cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
#else
  cv::cvtColor(img, img, CV_GRAY2BGR);
#endif

  // Plot tiles
  img.plotTiles();

  // Plot matches
  const cv::Scalar green(100, 255, 100);
  const unsigned int n = matches.size();
  for (unsigned int ii = 0; ii < n; ii++)
    img.plotFeature(matches[ii].current, green);

  std::string str = std::string("Matches: ") + std::to_string(n);

  cv::putText(img, str, cv::Point((int)10, (int)img.rows - 10),
              cv::FONT_HERSHEY_PLAIN, 1.0, green, 1.5, 8, false);
}

void Tracker::featureDetection(const TiledImage &img,
                               FeatureList &new_feature_list,
                               const double &timestamp,
                               unsigned int frame_number,
                               FeatureList &old_feature_list) {
  // Extract new features
  getFASTFeaturesPyramid(img_pyramid_, timestamp, frame_number,
                         new_feature_list, old_feature_list);
}

void Tracker::getImagePyramid(const TiledImage &img,
                              ImagePyramid &pyramid) const {
  // clear the vector of images
  pyramid.clear();

  // the first pyramid level is the highest resolution image
  pyramid.push_back(img);

  for (unsigned int i = 1; i < pyramid_depth_; i++)  // for all pyramid depths
  {
    // create a down-sampled image
    TiledImage downsampledImage;
    downsampledImage.setTileSize(img.getTileHeight(), img.getTileWidth());
    // reduce the image size by a factor of 2
    cv::pyrDown(pyramid[i - 1], downsampledImage);
    // add the down-sampled image to the pyramid
    pyramid.emplace_back(downsampledImage);
  }
}

void Tracker::getFASTFeaturesPyramid(ImagePyramid &pyramid,
                                     const double &timestamp,
                                     unsigned int frame_number,
                                     FeatureList &features,
                                     FeatureList &old_features) {
  // Loop through the image pyramid levels and get the fast features
  for (unsigned int l = 0; l < pyramid.size(); l++) {
    getFASTFeaturesImage(pyramid[l], timestamp, frame_number, l, features,
                         old_features);
  }
}

void Tracker::getFASTFeaturesImage(TiledImage &img, const double &timestamp,
                                   unsigned int frame_number,
                                   unsigned int pyramid_level,
                                   FeatureList &features,
                                   FeatureList &old_features) {
  Keypoints keypoints;  // vector of cv::Keypoint
  // scaling for points dependent on pyramid level to get their position
  double scale_factor = pow(2, static_cast<double>(pyramid_level));
#ifdef MULTI_UAV
  cv::FAST(img, keypoints, fast_detection_delta_, non_max_supp_);
  Descriptors descriptors;
  place_recognition_->compute(img, keypoints, descriptors,
                              fast_detection_delta_);
#else
  // Get FAST features
  cv::Mat descriptors;
  cv::FAST(img, keypoints, fast_detection_delta_, non_max_supp_);
#endif
  // Create a blocked mask which is 1 in the neighborhood of old features, and
  // 0 elsewhere.
  cv::Mat blocked_img = cv::Mat::zeros(img.rows, img.cols, CV_8U);
  computeNeighborhoodMask(old_features, img, blocked_img);

  // For all keypoints, extract the relevant information and put it into the
  // vector of Feature structs
  FeatureList candidate_features;
  float intensity = 0.f;
  for (int i = 0; i < keypoints.size(); i++) {
#ifdef PHOTOMETRIC_CALI
    intensity = computeIntensity(img, static_cast<int>(keypoints[i].pt.x),
                                 static_cast<int>(keypoints[i].pt.y));
#endif
    Feature feature(timestamp, frame_number,
                    static_cast<double>(keypoints[i].pt.x) * scale_factor,
                    static_cast<double>(keypoints[i].pt.y) * scale_factor,
                    pyramid_level, keypoints[i].response, intensity);
    // Check whether features fall in the bounds of allowed features; if
    // yes, add it to candidate features and store its designated bucket
    // information
    if (isFeatureInsideBorder(feature, img, margin_,
                              feature.getPyramidLevel())) {
#ifdef MULTI_UAV
      cv::Mat d = descriptors.row(i);
      assert(!d.empty());
      feature.setDescriptor(d.clone());
#endif
      candidate_features.push_back(feature);
    }
  }

  // Sort the candidate features by FAST score (highest first)
  std::sort(candidate_features.begin(), candidate_features.end(),
            [](const Feature &f1, const Feature &f2) -> bool {
              return f1.getFastScore() > f2.getFastScore();
            });

  // Append candidate features which are not neighbors of tracked features.
  // Note: old features are not supposed to be in here, see.
  // updateTrailLists()
  appendNonNeighborFeatures(img, features, candidate_features, blocked_img);
}

void Tracker::computeNeighborhoodMask(const FeatureList &features,
                                      const cv::Mat &img, cv::Mat &mask) const {
  // Loop through candidate features.
  // Create submask for each feature
  for (const auto &feature : features) {
    // Rounded feature coordinates
    double x = static_cast<int>(std::round(feature.getXDist()));
    double y = static_cast<int>(std::round(feature.getYDist()));

    // Submask top-left corner and width/height determination.
    int x_tl, y_tl, w, h;
    // Width
    if (x - block_half_length_ < 0) {
      x_tl = 0;
      w = block_half_length_ + 1 + x;
    } else if (x + block_half_length_ > img.cols - 1) {
      x_tl = x - block_half_length_;
      w = block_half_length_ + img.cols - x;
    } else {
      x_tl = x - block_half_length_;
      w = 2 * block_half_length_ + 1;
    }
    // Height
    if (y - block_half_length_ < 0) {
      y_tl = 0;
      h = block_half_length_ + 1 + y;
    } else if (y + block_half_length_ > img.rows - 1) {
      y_tl = y - block_half_length_;
      h = block_half_length_ + img.rows - y;
    } else {
      y_tl = y - block_half_length_;
      h = 2 * block_half_length_ + 1;
    }

    // Submask application
    cv::Mat blocked_box =
        mask(cv::Rect(x_tl, y_tl, w, h));  // box to block off in the image
    blocked_box.setTo(cv::Scalar(
        1));  // block out the surrounding area by setting the mask to 1
  }
}

bool Tracker::isFeatureInsideBorder(Feature &feature, const TiledImage &img,
                                    unsigned int margin,
                                    unsigned int pyramid_level) const {
  // Return true if the feature falls within the border at the specified
  // pyramid level
  static auto pow_exp = static_cast<double>(pyramid_level);
  static int xMin = static_cast<int>(margin);
  static int xMax =
      static_cast<int>(floor(img.cols / pow(2, pow_exp) - margin) - 1);
  static int yMin = static_cast<int>(margin);
  static int yMax =
      static_cast<int>(floor(img.rows / pow(2, pow_exp) - margin) - 1);
  PixelCoor feature_coord = {0, 0};
  getScaledPixelCoordinate(feature, feature_coord);
  return (feature_coord.x >= xMin) && (feature_coord.x <= xMax) &&
         (feature_coord.y >= yMin) && (feature_coord.y <= yMax);
}

void Tracker::getScaledPixelCoordinate(const Feature &feature,
                                       PixelCoor &scaled_coord) {
  // Transform the feature into scaled coordinates
  auto pow_exp = static_cast<double>(feature.getPyramidLevel());
  scaled_coord.x =
      static_cast<int>(std::round(feature.getXDist() / pow(2, pow_exp)));
  scaled_coord.y =
      static_cast<int>(std::round(feature.getYDist() / pow(2, pow_exp)));
}

void Tracker::appendNonNeighborFeatures(TiledImage &img, FeatureList &features,
                                        FeatureList &candidate_features,
                                        cv::Mat &mask) const {
  // Loop through candidate features.
  // If the patch is not blocked and there is room in the bucket, add it to
  // features
  for (auto &f : candidate_features) {
    double x = std::round(f.getXDist());  // approximate pixel coords (integer)
    double y = std::round(f.getYDist());  // approximate pixel coords (integer)

    // if the point is not blocked out
    if (mask.at<unsigned char>(y, x) == 0) {
      // if there is room in the bucket
      Feature &candidate_feature = f;
      img.setTileForFeature(candidate_feature);

      // add it to the features and update the bucket count
      features.push_back(f);

      // block out the surrounding area by setting the mask to 1
      cv::Mat blocked_box = mask(
          cv::Rect(x - block_half_length_, y - block_half_length_,
                   2 * block_half_length_ + 1, 2 * block_half_length_ + 1));
      blocked_box.setTo(cv::Scalar(1));
    }
  }
}

void Tracker::removeOverflowFeatures(TiledImage &img1, TiledImage &img2,
                                     FeatureList &features1,
                                     FeatureList &features2) const {
  img1.resetFeatureCounts();

  // Loop through current features to update their tile location
  for (auto i = features2.size(); i >= 1; i--) {
    img1.setTileForFeature(features1[i - 1]);
    img2.setTileForFeature(features2[i - 1]);

    img1.incrementFeatureCountAtTile(features1[i - 1].getTileRow(),
                                     features1[i - 1].getTileCol());
    img2.incrementFeatureCountAtTile(features2[i - 1].getTileRow(),
                                     features2[i - 1].getTileCol());
  }
  // Loop through current features to check if there are more than the max
  // allowed per tile and delete all features from this tile if so.
  // They are sorted by score so start at the bottom.
  for (int i = features2.size() - 1; i >= 1; i--) {
    const unsigned int feat2_row = features2[i - 1].getTileRow();
    const unsigned int feat2_col = features2[i - 1].getTileCol();
    const unsigned int count = img2.getFeatureCountAtTile(feat2_row, feat2_col);
    if (count > img1.getMaxFeatPerTile()) {
      // if the bucket is overflowing, remove the feature.
      features1.erase(features1.begin() + i - 1);
      features2.erase(features2.begin() + i - 1);
    }
  }
}

// this function automatically gets rid of points for which tracking fails
void Tracker::featureTracking(const cv::Mat &img1, const cv::Mat &img2,
                              const cv::Mat &img2_origin,
                              FeatureList &features1, FeatureList &features2,
                              const double &timestamp2,
                              const unsigned int frame_number2) const {
  // Convert features to OpenCV keypoints
  std::vector<cv::Point2f> pts1;
  pts1.resize(features1.size());
  for (unsigned int i = 0; i < pts1.size(); i++) {
    pts1[i] = features1[i].getDistPoint2f();
  }

  std::vector<uchar> status;
  std::vector<float> err;
  // Prevents calcOpticalFlowPyrLK crash
  std::vector<cv::Point2f> pts2;
  if (!pts1.empty()) {
#ifdef PHOTOMETRIC_CALI
    if (CALIBRATION_DONE) {
      cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err,
                               win_size_photo_, max_level_photo_, term_crit_,
                               cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
                               min_eig_thr_photo_);  // 0.001
    } else {
#endif
      cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, win_size_,
                               max_level_, term_crit_,
                               cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
                               min_eig_thr_);  // 0.001

#ifdef PHOTOMETRIC_CALI
    }
#endif
  }

  // Output features which KLT tracked and stayed inside the frame
  int index_correction = 0;
  float intensity = 0.f;
  for (unsigned int i = 0; i < status.size(); i++) {
    const cv::Point2f pt = pts2.at(i);
    if (status.at(i) != 0 && pt.x >= -0.5 && pt.y >= -0.5 &&
        pt.x <= img2.cols - 0.5 && pt.y <= img2.rows - 0.5) {
#ifdef PHOTOMETRIC_CALI
      intensity = computeIntensity(img2_origin, static_cast<int>(pts2[i].x),
                                   static_cast<int>(pts2[i].y));
#endif

      Feature new_feature(timestamp2, frame_number2, pt.x, pt.y,
                          features1[i - index_correction].getPyramidLevel(),
                          features1[i - index_correction].getFastScore(),
                          intensity);

#ifdef MULTI_UAV
      cv::Mat d = features1[i - index_correction].getDescriptor();
      assert(!d.empty());
      new_feature.setDescriptor(d);
#endif

      features2.push_back(new_feature);
    } else {
      features1.erase(features1.begin() + (i - index_correction));
      index_correction++;
    }
  }

  // Sanity check
  assert(features1.size() == features2.size());
}

/**
 * @brief Photometric calibration
 *
 */

#ifdef PHOTOMETRIC_CALI
void Tracker::refinePhotometricParams(const TrackList &tracks) {
  if (tracks.empty()) {
    return;
  }
  tracks_intensity_history_.clear();
  tracks_intensity_current_.clear();
  frame_diff_history_.clear();
  points_prev_.clear();
  points_curr_.clear();
  int tracks_counter =
      static_cast<int>(tracks.size());  // count the n of tracks
  int f = 1;                            // starting from the second last frame

  while (tracks_counter > 4) {  // required for RANSAC gain in photometric calib
    Intensities intensities_frame_f, intensities_current_f;
    std::vector<std::pair<int, int>> prev_point_frame_f, curr_point_curr_f;
    for (const Track &t : tracks) {
      if (static_cast<int>(t.size()) - 1 < f) {
        tracks_counter--;  // decrease the n of tracks
        continue;          // means that the track is not long enought
      }

      intensities_frame_f.emplace_back(
          t.at(t.size() - 1 - f)  // t.size()-1 is t.back()
              .getIntensity());   // starting from the back (namely the newest)

      // the intensities match that the current frame has with the frame f
      intensities_current_f.emplace_back(
          t.back().getIntensity());  // this is the current frame
      prev_point_frame_f.emplace_back(
          std::pair<int, int>(t.at(t.size() - 1 - f).getXDist(),
                              t.at(t.size() - 1 - f).getYDist()));
      curr_point_curr_f.emplace_back(
          std::pair<int, int>(static_cast<int>(t.back().getXDist()),
                              static_cast<int>(t.back().getYDist())));
    }

    if (intensities_frame_f.empty()) {
      continue;
    }

    points_prev_.push_back(prev_point_frame_f);
    points_curr_.push_back(curr_point_curr_f);
    tracks_intensity_history_.push_back(intensities_frame_f);
    tracks_intensity_current_.push_back(intensities_current_f);

    frame_diff_history_.push_back(f);
    f++;
  }

  calibrator_->ProcessCurrentFrame(
      tracks_intensity_history_, tracks_intensity_current_, frame_diff_history_,
      points_prev_, points_curr_, false);
  if (!CALIBRATION_DONE) {
    CALIBRATION_DONE = calibrator_->isReady();
  }
}

void Tracker::setIntensistyHistory(const TrackList &tracks) {
  boost::thread t(&Tracker::refinePhotometricParams, this, tracks);
  t.detach();
}

void Tracker::calibrateImage(TiledImage &img1, TiledImage &img2) {
  // 4 points neeed for photometric gains estiamtion
  if (previous_features_.empty() || previous_features_.size() < 4) {
    if (CALIBRATION_DONE) {
      calibrator_->getCorrectedImage(img2);
    }
    return;
  }
  // preparing points for photometric calibration
  std::vector<uchar> status;
  std::vector<float> err;

  std::vector<cv::Point2f> pts1, pts2;
  pts1.reserve(previous_features_.size());
#ifdef TIMING
  // Set up and start internal timing
  clock_t clock1, clock2, clock3, clock4, clock5;
  clock1 = clock();
#endif
  for (const auto &f : previous_features_) {
    pts1.emplace_back(f.getDistPoint2f());
  }

  cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, win_size_,
                           max_level_, term_crit_,
                           cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
                           min_eig_thr_);  // 0.001

#ifdef TIMING
  clock2 = clock();
  std::cout << "calcOpticalFlowPyrLK  ======================================= "
            << (double)(clock2 - clock1) / CLOCKS_PER_SEC * 1000 << " ms\n";
#endif
  // 4 points neeed for photometric gains estiamtion
  if (pts2.empty() || pts2.size() < 4) {
    if (CALIBRATION_DONE) calibrator_->getCorrectedImage(img2);
    return;
  }

  std::vector<float> intensities_prev, intensities_curr;
  std::vector<std::pair<int, int>> pair_prev, pair_curr;

  for (unsigned int i = 0; i < status.size(); i++) {
    const cv::Point2f &pt = pts2[i];
    if (status[i] && pt.x >= -0.5 && pt.y >= -0.5 && pt.x <= img2.cols - 0.5 &&
        pt.y <= img2.rows - 0.5) {
      intensities_curr.emplace_back(computeIntensity(
          img2, static_cast<int>(pt.x), static_cast<int>(pt.y)));

      intensities_prev.emplace_back(previous_features_[i].getIntensity());

      pair_prev.emplace_back(std::pair<int, int>(
          static_cast<int>(previous_features_[i].getXDist()),
          static_cast<int>(previous_features_[i].getYDist())));

      pair_curr.emplace_back(
          std::pair<int, int>(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }
  }
#ifdef TIMING
  clock3 = clock();
  std::cout << "emplace_back  ======================================= "
            << (double)(clock3 - clock2) / CLOCKS_PER_SEC * 1000 << " ms\n";
#endif
  // 4 points neeed for photometric gains estiamtion
  if (intensities_curr.size() < 4) {
    if (CALIBRATION_DONE) {
      calibrator_->getCorrectedImage(img2);
    }
    return;
  }
  static const std::vector<int> ind = {1};
  std::vector<std::vector<float>> i_intensities_prev = {intensities_prev};
  std::vector<std::vector<float>> i_intensities_curr = {intensities_curr};
  std::vector<std::vector<std::pair<int, int>>> i_pair_prev = {pair_prev};
  std::vector<std::vector<std::pair<int, int>>> i_pair_curr = {pair_curr};

  calibrator_->ProcessCurrentFrame(i_intensities_prev, i_intensities_curr, ind,
                                   i_pair_prev, i_pair_curr);

#ifdef TIMING
  clock4 = clock();
  std::cout << "ProcessCurrentFrame  ======================================= "
            << (double)(clock4 - clock3) / CLOCKS_PER_SEC * 1000 << " ms\n";
#endif
  calibrator_->getCorrectedImage(img2);

  fast_detection_delta_ = fast_detection_delta_photo_;
  CALIBRATION_DONE = true;

#ifdef TIMING
  clock5 = clock();
  std::cout << "end?!  ======================================= "
            << (double)(clock5 - clock4) / CLOCKS_PER_SEC * 1000 << " ms\n";
  std::cout << "TOT  ======================================= "
            << (double)(clock5 - clock1) / CLOCKS_PER_SEC * 1000 << " ms\n";
#endif
}

float Tracker::computeIntensity(const cv::Mat &img, const int x,
                                const int y) const {
  int counter = 0, x_win, y_win;
  float o = 0.f;
  const auto half_kernel = static_cast<int>(intensities_kernel_size_ / 2.0);

  for (int r = -half_kernel; r < half_kernel; r++) {
    for (int c = -half_kernel; c < half_kernel; c++) {
      x_win = x + c;
      y_win = y + r;
      if (x_win >= 0 && x_win < img.cols && y_win >= 0 && y_win < img.rows) {
        o += static_cast<float>(img.at<uchar>(y_win, x_win)) / 255.f;
        counter++;
      }
    }
  }
  return o / static_cast<float>(counter);
}

#endif

/**
 * @brief Multi UAV block
 *
 */

#ifdef MULTI_UAV

MsckfMatches &Tracker::getMsckfMatches() {
  return place_recognition_->getMsckfMatches();
}

SlamMatches &Tracker::getSlamMatches() {
  return place_recognition_->getSlamMatches();
}

void Tracker::updateOppMatches(const TrackList &current_msckf_tracks,
                               const TrackList &current_slam_tracks,
                               const TrackList &current_opp_tracks) {
  return place_recognition_->updateOppMatches(
      current_msckf_tracks, current_slam_tracks, current_opp_tracks);
}

void Tracker::cleanSlamMatches() { place_recognition_->cleanSlamMatches(); }

void Tracker::addKeyframe(const KeyframePtr &keyframe) {
  place_recognition_->addKeyframe(keyframe);
}

OppIDListPtr Tracker::getOppIds() { return place_recognition_->getOppIds(); };

#endif
