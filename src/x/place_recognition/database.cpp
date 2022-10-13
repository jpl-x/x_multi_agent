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

#include "x/place_recognition/database.h"

#include <boost/log/trivial.hpp>

using namespace x;

Database::Database(PRVocabularyPtr vocabulary_ptr, const double pr_score_thr)
    : vlad_(std::move(vocabulary_ptr)), pr_score_thr_(pr_score_thr) {}

VLADVec Database::computeVLAD(const cv::Mat &descriptors) {
  return vlad_.computeVLAD(descriptors);
}

void Database::findCandidate(const int uav_id, const VLADVec &query_vlad,
                             KeyframePtr &best_candidate) {
  double score = 0.f;
  for (auto &keyframe : keyframes_) {
    if (keyframe->findOtherUavId(uav_id)) {
      continue;
    }
    VLADVec train_vlad = keyframe->getVLAD();
    double score_tmp = vlad_.computeScore(query_vlad, train_vlad);
    if (score_tmp > pr_score_thr_ && score_tmp > score) {
      score = score_tmp;
      best_candidate = keyframe;
    }
  }
  if (best_candidate != nullptr) {
    best_candidate->setOtherUavId(uav_id);
  }
  BOOST_LOG_TRIVIAL(debug) << "Score place recog : " << score
                           << ", Keyframe database size: " << keyframes_.size()
                           << std::endl;
}

void Database::addKeyframe(const KeyframePtr &keyframe) {
  cv::Mat desc = keyframe->getDescriptors();
  VLADVec vlad = vlad_.computeVLAD(desc);
  keyframe->setVLAD(vlad);
  keyframes_.emplace_back(keyframe);
  if (keyframes_.size() > max_keyframe_number_) {
    keyframes_.erase(keyframes_.begin());
  }
  BOOST_LOG_TRIVIAL(debug) << "New keyframe inserted! Keyframe size : "
                           << keyframes_.size() << std::endl;
}