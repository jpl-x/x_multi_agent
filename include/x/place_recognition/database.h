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

#if !defined(X_DATABASE_H) && defined(MULTI_UAV)
#define X_DATABASE_H

#include <deque>

#include "x/place_recognition/keyframe.h"
#include "x/place_recognition/types.h"
#include "x/place_recognition/vlad.h"

namespace x {
/**
 * @brief The database contains the keyframes and a pointer to the vocabulary.
 *
 */
class Database {
 public:
  /**
   * Database constructor, needs vocabulary and a score threshold to find the
   * keyframe matches
   *
   * @param vocabulary_ptr
   * @param pr_score_thr
   */
  Database(PRVocabularyPtr vocabulary_ptr, double pr_score_thr);

  /**
   * Add keyframe to the database
   *
   * @param keyframe
   */
  void addKeyframe(const KeyframePtr &keyframe);

  /**
   * Find a keyframe candidate using the received VLAD
   *
   * @param[in] uav_id
   * @param[in] query_vlad
   * @param[out] best_candidate
   */
  void findCandidate(int uav_id, const VLADVec &query_vlad,
                     KeyframePtr &best_candidate);

  /**
   * Compute the VLAD out of the descriptors
   *
   * @param descriptors
   * @return
   */
  VLADVec computeVLAD(const cv::Mat &descriptors);
  
 private:
  size_t max_keyframe_number_ = 15;

  VLAD vlad_;

  std::vector<KeyframePtr> keyframes_;

  const double pr_score_thr_;
};
}  // namespace x

#endif