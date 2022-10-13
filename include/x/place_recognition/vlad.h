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

#if !defined(X_VLAD_H) && defined(MULTI_UAV)
#define X_VLAD_H

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

#include "x/place_recognition/types.h"
#include "x/vision/feature.h"
#include "x/vision/track.h"
#include "x/vision/types.h"

namespace x {

class VLAD {
 public:
  /**
   * Constructor, needs a binary vocabulary for ORB descriptors
   *
   * @param vocabulary
   */
  explicit VLAD(PRVocabularyPtr vocabulary);

  /**
   *
   * @param x is the concatenation of the ORB descriptors extracted from the
   * incoming image
   *
   * @return the computed binary VLAD descriptor
   */
  [[nodiscard]] VLADVec computeVLAD(const cv::Mat &x);

  /**
   * Compute the similarity score between two VLADs
   * @param x VLAD
   * @param y VLAD
   * @return computed score
   */
  [[nodiscard]] double computeScore(const VLADVec &x, const VLADVec &y) const;

 private:
  /**
   * Retrieve closest centroid to the selected descriptor
   *
   * @param[in] desc
   * @param[out] centroid
   * @param[out] id_centroid
   */
  void findCentroid(const cv::Mat &desc, cv::Mat &centroid,
                    unsigned int &id_centroid);

  PRVocabularyPtr vocabulary_;

  const int d_length_;
  const int clusters_n_;
  const int v_length_;
};

}  // namespace x

#endif