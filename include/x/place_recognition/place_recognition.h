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

#if !defined(X_PLACE_RECOGNITION_H) && defined(MULTI_UAV)
#define X_PLACE_RECOGNITION_H

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <thread>

#include "x/ekf/simple_state.h"
#include "x/place_recognition/database.h"
#include "x/place_recognition/keyframe.h"
#include "x/place_recognition/types.h"
#include "x/vision/camera.h"
#include "x/vision/types.h"

namespace x {

namespace fsm = std::filesystem;

class PlaceRecognition {
 public:
  /**
   * Place recognition constructor
   *
   * @param cam
   * @param patch_size
   * @param scale_factor
   * @param pyramid_levels
   * @param vocabulary_path
   * @param fast_thr
   * @param desc_type
   * @param pr_score_thr
   * @param pr_min_distance
   * @param pr_ratio_thr
   */
  PlaceRecognition(Camera cam, int patch_size, float scale_factor,
                   int pyramid_levels, const fsm::path &vocabulary_path,
                   int fast_thr, int desc_type, double pr_score_thr,
                   double pr_min_distance, double pr_ratio_thr);

  /**
   *
   *
   * @param img
   * @param keypoints
   * @param descriptors
   * @param fast_thr
   */
  void compute(const cv::Mat &img, Keypoints &keypoints,
               Descriptors &descriptors, int fast_thr);

  void findPlace(int receiver_uav_id, const cv::Mat &descriptors,
                 KeyframePtr &candidate);

  bool findCorrespondences(int uav_id, const TrackList &current_msckf_tracks,
                           const TrackListPtr &received_msckf_tracks_ptr,
                           const TrackList &current_slam_tracks,
                           const TrackListPtr &received_slam_tracks_ptr,
                           const TrackList &current_opp_tracks,
                           const TrackListPtr &received_opp_tracks_ptr,
                           const std::shared_ptr<SimpleState> &state_ptr,
                           cv::Mat &src, cv::Mat &dst);

  /** Getter **/
  cv::Mat getVisualMatchesImage() { return matches_img_; }

  SlamMatches &getSlamMatches();
  MsckfMatches &getMsckfMatches();

  OppIDListPtr getOppIds() { return opp_matches_ids_; };

  void retrievePlace();
  // void getPRfeatures(PRfeatures& pr_features) { pr_features = pr_features_; }

  void cleanSlamMatches() { slam_matches_.clear(); }

  /**
   * @brief The opportunistic matches are the matches between a Opp tracks on
   * the current UAV and the received data. Here is were MSCKF-MSCKF matches can
   * happen.
   *
   * @param current_msckf_tracks tracks that have been "upgraded" to MSCKF
   * tracks.
   * @param current_slam_tracks tracks that have been "upgraded" to SLAM tracks.
   * @param current_opp_tracks tracks that are still in the system.
   */
  void updateOppMatches(const TrackList &current_msckf_tracks,
                        const TrackList &current_slam_tracks,
                        const TrackList &current_opp_tracks);

  void addKeyframe(const KeyframePtr &frame);

  cv::Mat getDescriptors();

 private:
  Camera camera_;
  cv::Ptr<cv::ORB> detector_;
  // cv::Ptr<cv::FeatureDetector> feature_detector_;
  // cv::Ptr<cv::DescriptorExtractor> descriptor_extractor_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  std::vector<std::vector<cv::DMatch>> matches_;

  cv::Mat matches_img_;
  MsckfMatches msckf_matches_;
  SlamMatches slam_matches_;
  OppMatches opp_matches_;
  OppIDListPtr opp_matches_ids_;
  MsckfSlamMatches hybrid_matches_;
  // PRfeatures pr_features_;

  double pr_min_distance_ = 60.0;
  double pr_ratio_thr_ = 0.89;

  Matrix H_, R_;
  Vector3 T_;

  void drawMatches(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point2f> &kpts1,
                   std::vector<cv::Point2f> &kpts2, std::vector<uchar> mask);
  void drawMatches(cv::Mat &src, cv::Mat &dst, Keypoints &kpts1,
                   Keypoints &kpts2, const std::vector<cv::DMatch> &m);

  // FBoW
  Descriptors descriptros_;
  DatabasePtr database_;
  PRVocabularyPtr vocabulary_;

  std::mutex mtx_;
};

}  // namespace x

#endif