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

#include "x/place_recognition/place_recognition.h"

#include <boost/log/trivial.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

using namespace x;

PlaceRecognition::PlaceRecognition(
    Camera cam, const int patch_size, const float scale_factor,
    const int pyramid_levels, const fsm::path &vocabulary_path,
    const int fast_thr, const int desc_type, const double pr_score_thr,
    const double pr_min_distance, const double pr_ratio_thr)
    : camera_{std::move(cam)},
      pr_min_distance_(pr_min_distance),
      pr_ratio_thr_(pr_ratio_thr) {
  opp_matches_ids_ = std::make_shared<std::vector<unsigned long long>>();
  // BoW
  if (vocabulary_path.empty() || !fsm::exists(vocabulary_path)) {
    throw std::invalid_argument(
        "You must provide an ORB vocabulary in the config file.");
  }
  BOOST_LOG_TRIVIAL(info)
      << "Loading ORB vocabulary... this could take a while..."
      << vocabulary_path << std::endl;

  vocabulary_ = std::make_shared<PRVocabulary>();
  vocabulary_->load(vocabulary_path);
  BOOST_LOG_TRIVIAL(info) << "Vocabulary read successfuly." << std::endl
                          << "Vocabulary descriptor: "
                          << vocabulary_->getDescritorType() << std::endl
                          << "Vocabulary descriptor size: "
                          << vocabulary_->getDescritorSize() << std::endl
                          << "K: " << vocabulary_->getBranchingFactor()
                          << std::endl
                          << "L: " << vocabulary_->getDepthLevels() << std::endl
                          << "size: " << vocabulary_->size() << std::endl;

  database_ = std::make_shared<Database>(vocabulary_, pr_score_thr);

  if (desc_type == DESCRIPTOR_TYPE::ORB) {
    detector_ = cv::ORB::create(500, scale_factor, pyramid_levels, 31, 0, 2,
                                cv::ORB::FAST_SCORE, patch_size, fast_thr);
  }

  // One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2
  // norms are preferable choices for SIFT and SURF descriptors,
  // NORM_HAMMING should be used with ORB, BRISK and BRIEF,
  // NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see
  // ORB::ORB constructor description).}
  matcher_ = cv::DescriptorMatcher::create(
      cv::DescriptorMatcher::BRUTEFORCE_HAMMING);  // by dafault WTA_K=2
}

void PlaceRecognition::compute(const cv::Mat &img, Keypoints &keypoints,
                               Descriptors &descriptors, const int fast_thr) {
#ifdef PHOTOMETRIC_CALI
  if (detector_->getFastThreshold() != fast_thr) {
    detector_->setFastThreshold(static_cast<int>(fast_thr));
  }
#endif
  /**
   * READ HERE :
   * https://stackoverflow.com/questions/61957611/missing-keypoints-from-image
   *
   * From OpenCV documentation:
   * keypoints – Input collection of keypoints. Keypoints for which a descriptor
   * cannot be computed are removed. Sometimes new keypoints can be added, for
   * example: SIFT duplicates keypoint with several dominant orientations (for
   * each orientation).
   *
   */

  detector_->compute(img, keypoints, descriptors);
  descriptors.copyTo(descriptros_);  // store a copy to send to the other UAVs
  assert(keypoints.size() == descriptors.rows);
}

void PlaceRecognition::drawMatches(cv::Mat &src, cv::Mat &dst,
                                   std::vector<cv::Point2f> &kpts1,
                                   std::vector<cv::Point2f> &kpts2,
                                   std::vector<uchar> mask) {
  Keypoints n_kpts1, n_kpts2;
  for (const cv::Point2f &kp : kpts1) {
    n_kpts1.emplace_back(camera_.denormalize(cv::KeyPoint(kp, 1.f)));
  }

  for (const cv::Point2f &kp : kpts2) {
    n_kpts2.emplace_back(camera_.denormalize(cv::KeyPoint(kp, 1.f)));
  }

  std::vector<cv::DMatch> m;
  for (int i = 0; i < static_cast<int>(mask.size()); i++) {
    if (mask[i]) {
      m.emplace_back(cv::DMatch(i, i, 0.f));
    }
  }

  cv::drawMatches(src, n_kpts1, dst, n_kpts2, m, matches_img_);
  dst = matches_img_;
}

void PlaceRecognition::drawMatches(cv::Mat &src, cv::Mat &dst, Keypoints &kpts1,
                                   Keypoints &kpts2,
                                   const std::vector<cv::DMatch> &m) {
  Keypoints n_kpts1, n_kpts2;
  for (const cv::KeyPoint &kp : kpts1) {
    n_kpts1.emplace_back(camera_.denormalize(kp));
  }
  for (const cv::KeyPoint &kp : kpts2) {
    n_kpts2.emplace_back(camera_.denormalize(kp));
  }

  cv::drawMatches(dst, n_kpts2, src, n_kpts1, m, matches_img_);
  cv::imshow("TEST MATCHES", matches_img_);
  cv::waitKey(0);
  // dst = matches_img_;
}

bool PlaceRecognition::findCorrespondences(
    const int uav_id, const TrackList &current_msckf_tracks,
    const TrackListPtr &received_msckf_tracks_ptr,
    const TrackList &current_slam_tracks,
    const TrackListPtr &received_slam_tracks_ptr,
    const TrackList &current_opp_tracks,
    const TrackListPtr &received_opp_tracks_ptr,
    const std::shared_ptr<SimpleState> &state_ptr, cv::Mat &src, cv::Mat &dst) {
  // NEDED BY THE F MATRIX
  std::vector<cv::Point2f> current_points;
  std::vector<cv::Point2f> received_points;

  // to store the matches
  MsckfMatches tmp_msckf_matches = MsckfMatches();
  SlamMatches tmp_slam_matches = SlamMatches();
  OppMatches tmp_opp_matches = OppMatches();
  // MsckfSlamMatches tmp_hybrid_matches = MsckfSlamMatches();

#ifndef GT_DEBUG

  const int MAX_CURR_MSCKF_ID = static_cast<int>(current_msckf_tracks.size());
  const int MAX_CURR_SLAM_ID =
      static_cast<int>(current_slam_tracks.size() + MAX_CURR_MSCKF_ID);
  const int MAX_REC_MSCKF_ID =
      static_cast<int>(received_msckf_tracks_ptr.size());
  const int MAX_REC_SLAM_ID =
      static_cast<int>(received_slam_tracks_ptr.size() + MAX_REC_MSCKF_ID);
  /**
   * @brief Create the PR feature structure
   *
   * The order of the resulting structure is:
   * MSCKF
   * SLAM
   * OPP
   *
   */
  // Current
  PRfeatures current_pr_features;
  // msckf
  std::for_each(
      current_msckf_tracks.begin(), current_msckf_tracks.end(),
      [&current_pr_features](const Track &t) {
        current_pr_features.descriptors.push_back(t.back().getDescriptor());
        current_pr_features.keypoints.emplace_back(
            cv::KeyPoint(static_cast<float>(t.back().getX()),
                         static_cast<float>(t.back().getY()), 1.0f));
      });
  // slam
  std::for_each(
      current_slam_tracks.begin(), current_slam_tracks.end(),
      [&current_pr_features](const Track &t) {
        current_pr_features.descriptors.push_back(t.back().getDescriptor());
        current_pr_features.keypoints.emplace_back(
            cv::KeyPoint(static_cast<float>(t.back().getX()),
                         static_cast<float>(t.back().getY()), 1.0f));
      });
  // opp
  std::for_each(
      current_opp_tracks.begin(), current_opp_tracks.end(),
      [&current_pr_features](const Track &t) {
        if (t.empty()) {
          throw std::runtime_error(
              "Opp track empty");  // TODO: HOW IS IT POSSIBLE THAT A
                                   // TRACK IS EMPTY?
        }
        cv::Mat desc = t.back().getDescriptor();
        current_pr_features.descriptors.push_back(t.back().getDescriptor());
        current_pr_features.keypoints.emplace_back(
            cv::KeyPoint(static_cast<float>(t.back().getX()),
                         static_cast<float>(t.back().getY()), 1.0f));
      });

  if (current_pr_features.empty()) {
    return false;
  }

  // Received
  PRfeatures received_pr_features;
  // msckf
  std::for_each(
      received_msckf_tracks_ptr.begin(), received_msckf_tracks_ptr.end(),
      [&received_pr_features](const TrackPtr &t) {
        received_pr_features.descriptors.push_back(t->back().getDescriptor());
        received_pr_features.keypoints.emplace_back(
            cv::KeyPoint(static_cast<float>(t->back().getX()),
                         static_cast<float>(t->back().getY()), 1.0f));
      });
  // slam
  std::for_each(received_slam_tracks_ptr.begin(),
                received_slam_tracks_ptr.end(),
                [&received_pr_features](const TrackPtr &t) {
                  cv::Mat desc = t->getDescriptor();
                  received_pr_features.descriptors.push_back(desc);
                  received_pr_features.keypoints.emplace_back(
                      cv::KeyPoint(static_cast<float>(t->back().getX()),
                                   static_cast<float>(t->back().getY()), 1.0f));
                });
  // opp
  std::for_each(received_opp_tracks_ptr.begin(), received_opp_tracks_ptr.end(),
                [&received_pr_features](const TrackPtr &t) {
                  cv::Mat desc = t->getDescriptor();
                  received_pr_features.descriptors.push_back(desc);
                  received_pr_features.keypoints.emplace_back(
                      cv::KeyPoint(static_cast<float>(t->back().getX()),
                                   static_cast<float>(t->back().getY()), 1.0f));
                });

  if (received_pr_features.empty()) {
    return false;
  }

  matches_.clear();
  matcher_->knnMatch(received_pr_features.descriptors,
                     current_pr_features.descriptors, matches_, 2);

  std::vector<cv::DMatch> good_matches;
  for (auto m : matches_) {
    if (m[0].distance < pr_min_distance_ &&
        m[0].distance < m[1].distance * pr_ratio_thr_) {
      good_matches.push_back(m[0]);
      // NEEDED BY THE F MATRIX
      received_points.emplace_back(
          cv::Point2f(received_pr_features.keypoints[m[0].queryIdx].pt));
      current_points.emplace_back(
          cv::Point2f(current_pr_features.keypoints[m[0].trainIdx].pt));
    }
  }

  if (good_matches.empty()) {
    return false;
  }

  std::vector<uchar> mask_22;
  cv::Mat F_1 = cv::findEssentialMat(current_points, received_points,
                                     camera_.getCVCameraMatrix(), cv::RANSAC,
                                     0.99, 1.0, mask_22);

  if (F_1.empty()) {
    return false;
  }
  int corr_id = 0;
  for (size_t i = 0; i < mask_22.size(); i++) {
    if (!mask_22[i]) {
      good_matches.erase(good_matches.begin() + static_cast<long>(i) - corr_id);
      corr_id++;
    }
  }

  // remove matches that refer to the same feature.
  std::vector<int> remove_ids;
  for (size_t i = 0; i < good_matches.size(); i++) {
    for (size_t j = i; j < good_matches.size(); j++) {
      // remove odd numbers
      if (i != j && (good_matches[i].queryIdx == good_matches[j].queryIdx ||
                     good_matches[i].trainIdx == good_matches[j].trainIdx)) {
        // std::cout << "DUPLICATO\n";
        remove_ids.push_back(static_cast<int>(j));
        break;
      }
    }
  }
  corr_id = 0;
  for (const int remove_id : remove_ids) {
    good_matches.erase(good_matches.begin() + remove_id - corr_id);
    corr_id++;
  }

  if (good_matches.empty()) {
    return false;
  }

  // The train is the current, the query is the sender
  // iterate over the good matches and create the match objects for the
  // collaborative updates
  for (auto &good_match : good_matches) {
    // REC - CURR

    // MSKCF Received
    if (good_match.queryIdx < MAX_REC_MSCKF_ID) {
      // MSKCF-OPP
      if (good_match.trainIdx >= MAX_CURR_SLAM_ID) {
#ifdef DEBUG
        std::cout << "\033[1;31mMatch MSCKF-OPP\033[0m" << std::endl;
#endif
        // we store the id of this track in such a way that when the track
        // manager updates the tracks, this opp track is converted to an MSCKF
        // track.
        opp_matches_ids_->push_back(
            current_opp_tracks[good_match.trainIdx - MAX_CURR_SLAM_ID].getId());

        // MSCKF-Opp
        tmp_msckf_matches.emplace_back(MsckfMatch(
            uav_id,
            current_opp_tracks[good_match.trainIdx - MAX_CURR_SLAM_ID].getId(),
            received_msckf_tracks_ptr[good_match.queryIdx]->getId(),
            received_msckf_tracks_ptr[good_match.queryIdx], state_ptr));
      }
    }

    // REC SLAM
    if (good_match.queryIdx >= MAX_REC_MSCKF_ID &&
        good_match.queryIdx < MAX_REC_SLAM_ID) {
      // SLAM - SLAM
      if (good_match.trainIdx >= MAX_CURR_MSCKF_ID &&
          good_match.trainIdx < MAX_CURR_SLAM_ID) {
        // SLAM-SLAM MATCH
        tmp_slam_matches.emplace_back(
            SlamMatch(uav_id, good_match.trainIdx - MAX_CURR_MSCKF_ID,
                      good_match.queryIdx - MAX_REC_MSCKF_ID, state_ptr));
      }

      // SLAM - OPP
      if (good_match.trainIdx >= MAX_CURR_SLAM_ID) {
        // we need to store this match , since the OPP track can potentially be
        // a SLAM feature!
        tmp_opp_matches.emplace_back(OppMatch(
            uav_id,
            current_opp_tracks[good_match.trainIdx - MAX_CURR_SLAM_ID].getId(),
            received_slam_tracks_ptr[good_match.queryIdx - MAX_REC_MSCKF_ID]
                ->getId(),
            good_match.queryIdx - MAX_REC_MSCKF_ID,
            received_slam_tracks_ptr[good_match.queryIdx - MAX_REC_MSCKF_ID],
            state_ptr, MatchType::SLAM));
      }
    }

    // REC OPP
    if (good_match.queryIdx >= MAX_REC_SLAM_ID) {
      // Opp-Opp
      if (good_match.trainIdx >= MAX_CURR_SLAM_ID) {
        // OPP-OPP
        unsigned long long idc =
            current_opp_tracks[good_match.trainIdx - MAX_CURR_SLAM_ID].getId();

        opp_matches_ids_->push_back(idc);

        unsigned long long idr =
            received_opp_tracks_ptr[good_match.queryIdx - MAX_REC_SLAM_ID]
                ->getId();

        tmp_msckf_matches.emplace_back(MsckfMatch(
            uav_id, idc, idr,
            received_opp_tracks_ptr[good_match.queryIdx - MAX_REC_SLAM_ID],
            state_ptr));
      }
    }
  }

#endif

#ifdef GT_DEBUG
  /**
   * CURRENT SLAM
   */
  for (size_t i = 0; i < current_slam_tracks.size(); i++) {
    Vector3 current_l = current_slam_tracks[i].back().getLandmark();
    for (size_t j = 0; j < received_msckf_tracks_ptr.size(); j++) {
      Vector3 received_l = received_msckf_tracks_ptr[j]->back().getLandmark();
      Vector3 diff = current_l - received_l;

      if (std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <
          0.001) {
        // TODO add support for hybrid matches
        // tmp_hybrid_matches.emplace_back(
        //     MsckfSlamMatch(uav_id, i, received_msckf_tracks_ptr[j],
        //     state_ptr));
      }
    }
  }

  for (size_t i = 0; i < current_slam_tracks.size(); i++) {
    Vector3 current_l = current_slam_tracks[i].back().getLandmark();
    for (size_t j = 0; j < received_slam_tracks_ptr.size(); j++) {
      Vector3 received_l = received_slam_tracks_ptr[j]->back().getLandmark();
      Vector3 diff = current_l - received_l;

      if (std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <
          0.001) {
        tmp_slam_matches.emplace_back(SlamMatch(uav_id, i, j, state_ptr));
      }
    }
  }
  for (size_t i = 0; i < current_slam_tracks.size(); i++) {
    Vector3 current_l = current_slam_tracks[i].back().getLandmark();
    for (size_t j = 0; j < received_opp_tracks_ptr.size(); j++) {
      Vector3 received_l = received_opp_tracks_ptr[j]->back().getLandmark();
      Vector3 diff = current_l - received_l;

      if (std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <
          0.001) {
        // std::cout << "MATCH OPP-SLAM" << std::endl;
        // TODO add support for hybrid matches
        // tmp_hybrid_matches.emplace_back(
        //     MsckfSlamMatch(uav_id, i, received_opp_tracks_ptr[j],
        //     state_ptr));
      }
    }
  }

  /**
   * CURRENT OPP
   */
  for (size_t i = 0; i < current_opp_tracks.size(); i++) {
    Vector3 current_l = current_opp_tracks[i].back().getLandmark();
    for (size_t j = 0; j < received_msckf_tracks_ptr.size(); j++) {
      Vector3 received_l = received_msckf_tracks_ptr[j]->back().getLandmark();
      Vector3 diff = current_l - received_l;

      if (std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <
          1e-6) {
        // std::cout << "MATCH! MSCKF-OPP" << std::endl;
        // wait that the opp tracks is upgraded or is lost
        // if lost-> MSCKF-MSCKF match
        // if upgraded -> MSCKF- SLAM/MSCKF match

        // This can be seen as an MSCKF-MSCKF match <--------------------

        unsigned long long idc = current_opp_tracks[i].getId();
        opp_matches_ids_->push_back(
            idc);  // these are the ids of the opp tracks that will be converted
                   // to msckf tracks

        unsigned long long idr = received_msckf_tracks_ptr[j]->getId();

        tmp_msckf_matches.emplace_back(MsckfMatch(
            uav_id, idc, idr, received_msckf_tracks_ptr[j], state_ptr));

        // current_points.emplace_back(current_keypoints[i].pt);
        // received_points.emplace_back(received_keypoints[j].pt);
        // MSCKF-MSCKF MATCH
        // tmp_opp_matches.emplace_back(OppMatch(
        //     uav_id, current_opp_tracks[i].getId(),
        //     received_msckf_tracks_ptr[j]->getId(), j,
        //     received_msckf_tracks_ptr[j], state_ptr, MatchType::MSCKF));
      }
    }
  }

  for (size_t i = 0; i < current_opp_tracks.size(); i++) {
    Vector3 current_l = current_opp_tracks[i].back().getLandmark();
    for (size_t j = 0; j < received_slam_tracks_ptr.size(); j++) {
      Vector3 received_l = received_slam_tracks_ptr[j]->back().getLandmark();
      Vector3 diff = current_l - received_l;

      // std::cout << "MATCH SLAM-OPP" << std::endl;

      if (std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <
          0.001) {
        tmp_opp_matches.emplace_back(
            OppMatch(uav_id, current_opp_tracks[i].getId(),
                     received_slam_tracks_ptr[j]->getId(), j,
                     received_slam_tracks_ptr[j], state_ptr, MatchType::SLAM));
      }
    }
  }
  for (size_t i = 0; i < current_opp_tracks.size(); i++) {
    Vector3 current_l = current_opp_tracks[i].back().getLandmark();
    for (size_t j = 0; j < received_opp_tracks_ptr.size(); j++) {
      Vector3 received_l = received_opp_tracks_ptr[j]->back().getLandmark();
      Vector3 diff = current_l - received_l;

      if (std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <
          1e-6) {
        // std::cout << "MATCH! OPP-OPP" << std::endl;
        // OPP-OPP MATCH
        // This can be seen as an MSCKF-MSCKF match.

        unsigned long long idc = current_opp_tracks[i].getId();
        opp_matches_ids_->push_back(
            idc);  // these are the ids of the opp tracks that will be converted
                   // to msckf tracks

        unsigned long long idr = received_opp_tracks_ptr[j]->getId();

        tmp_msckf_matches.emplace_back(MsckfMatch(
            uav_id, idc, idr, received_opp_tracks_ptr[j], state_ptr));
      }
    }
  }

#endif
  slam_matches_ = tmp_slam_matches;

  /**
   * This for loop can be done before, to check if we already know about the
   * track we received can be a prior on the track matches.
   */

  int corrected_index = 0;
  for (MsckfMatch &t : msckf_matches_) {
    corrected_index = 0;
    for (size_t i = 0; i < tmp_msckf_matches.size(); i++) {
      if (t.uav_id == tmp_msckf_matches[i - corrected_index].uav_id &&
          t.id_received_track ==
              tmp_msckf_matches[i - corrected_index].id_received_track) {
        t.received_track_ptr =
            tmp_msckf_matches[i - corrected_index].received_track_ptr;
        t.state = tmp_msckf_matches[i - corrected_index].state;

        tmp_msckf_matches.erase(
            tmp_msckf_matches.begin() + static_cast<long>(i) -
            corrected_index);  // delete the track in the tmp,
                               // in this way we don't insert
                               // more times the same track
        // correct the index after erasing the match
        corrected_index++;
        break;
      }
    }
  }
  // update the msckf matches container with the new matches.
  msckf_matches_.insert(msckf_matches_.end(), tmp_msckf_matches.begin(),
                        tmp_msckf_matches.end());

  for (OppMatch &t : opp_matches_) {
    corrected_index = 0;
    for (size_t i = 0; i < tmp_opp_matches.size(); i++) {
      if (t.uav_id == tmp_opp_matches[i - corrected_index].uav_id &&
          t.id_received_track ==
              tmp_opp_matches[i - corrected_index].id_received_track &&
          tmp_opp_matches[i - corrected_index].type == MatchType::MSCKF) {
        t.received_track_ptr =
            tmp_opp_matches[i - corrected_index].received_track_ptr;
        t.state = tmp_opp_matches[i - corrected_index].state;

        tmp_opp_matches.erase(tmp_opp_matches.begin() + static_cast<long>(i) -
                              corrected_index);  // delete the track in the tmp,
                                                 // in this way we don't insert
                                                 // more times the same track
        // correct the index after erasing the match
        corrected_index++;
        break;
      }
    }
  }

  opp_matches_.insert(opp_matches_.end(), tmp_opp_matches.begin(),
                      tmp_opp_matches.end());

  return !msckf_matches_.empty() || !slam_matches_.empty() ||
         !opp_matches_.empty();
}

MsckfMatches &PlaceRecognition::getMsckfMatches() { return msckf_matches_; }

SlamMatches &PlaceRecognition::getSlamMatches() { return slam_matches_; }

void PlaceRecognition::updateOppMatches(const TrackList &current_msckf_tracks,
                                        const TrackList &current_slam_tracks,
                                        const TrackList &current_opp_tracks) {
  bool found = false;
  int corrected_index = 0;
  for (size_t i = 0; i < opp_matches_.size(); i++) {
    uniqueId match_id =
        opp_matches_[i]
            .id_current_track;  // id of the track matched from the current UAV

    found = false;
    for (const auto &current_msckf_track : current_msckf_tracks) {
      if (current_msckf_track.getId() == match_id) {
        if (opp_matches_[i].type == MatchType::MSCKF) {  // the other UAV MSCKF
#ifdef VERBOSE
          std::cout << "\033[1;31mUpgrade Opp match to MSCKF-MSCKF\033[0m"
                    << std::endl;
#endif
          msckf_matches_.emplace_back(MsckfMatch(
              opp_matches_[i].uav_id, current_msckf_track.getId(),
              opp_matches_[i].id_received_track,
              opp_matches_[i].received_track_ptr, opp_matches_[i].state));
        } else {
          // this would be a SLAM-MSCKF match
        }
        opp_matches_.erase(opp_matches_.begin() + static_cast<long>(i) -
                           corrected_index);
        found = true;
        corrected_index++;
        break;
      }
    }
    for (size_t j = 0; j < current_slam_tracks.size() && !found; j++) {
      if (current_slam_tracks[j].getId() == match_id) {
        if (opp_matches_[i].type == MatchType::MSCKF) {  // the other UAV MSCKF
          // this would be a MSCKF-SLAM match
        } else {
#ifdef VERBOSE
          std::cout << "\033[1;31mUpgrade Opp match to SLAM-SLAM\033[0m"
                    << std::endl;
          // std::cout << "\033[1;31mCurrent Feature ID : \033[0m" << j
          //           << "\033[1;31m, Received Feature ID : \033[0m"
          //           << opp_matches_[i].received_feature_id << std::endl;
#endif
          slam_matches_.emplace_back(SlamMatch(
              opp_matches_[i].uav_id,
              static_cast<int>(
                  j),  // It's no opp_matches[i].current_feature_id because this
                       // can change accoridng to the feature state update.
              opp_matches_[i].received_feature_id, opp_matches_[i].state));
        }
        opp_matches_.erase(opp_matches_.begin() + static_cast<long>(i) -
                           corrected_index);
        corrected_index++;
        break;
      }
    }
  }

  /**
   * Discard Tracks that are not tracked anymore
   */
  corrected_index = 0;
  for (size_t i = 0; i < opp_matches_.size(); i++) {
    uniqueId stored_track = opp_matches_[i - corrected_index].id_current_track;
    found = false;
    for (const auto &current_opp_track : current_opp_tracks) {
      if (stored_track == current_opp_track.getId()) {
        found = true;
        break;
      }
    }
    if (!found) {
#ifdef DEBUG
      // std::cout << "\033[1;31mDeleting opp match ...\033[0m" << std::endl;
#endif
      opp_matches_.erase(opp_matches_.begin() + static_cast<long>(i) -
                         corrected_index);
      corrected_index++;
    }
  }

  for (const auto &current_opp_track : current_opp_tracks) {
    for (size_t i = 0; i < opp_matches_ids_->size(); i++) {
      if (current_opp_track.getId() == opp_matches_ids_->at(i)) {
        opp_matches_ids_->erase(opp_matches_ids_->begin() +
                                static_cast<long>(i));
        break;
      }
    }
  }
}

void PlaceRecognition::findPlace(const int receiver_uav_id,
                                 const cv::Mat &descriptors,
                                 KeyframePtr &candidate) {
  // if it finds a candidate then send a message
  std::lock_guard<std::mutex> lck(mtx_);
  database_->findCandidate(receiver_uav_id, descriptors, candidate);
}

void PlaceRecognition::addKeyframe(const KeyframePtr &frame) {
  std::lock_guard<std::mutex> lck(mtx_);
  database_->addKeyframe(frame);
}

cv::Mat PlaceRecognition::getDescriptors() {
  return database_->computeVLAD(descriptros_);
}
