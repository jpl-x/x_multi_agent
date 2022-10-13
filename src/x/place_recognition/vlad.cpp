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

#include "x/place_recognition/vlad.h"

#include <boost/log/trivial.hpp>
#include <utility>

using namespace x;

VLAD::VLAD(PRVocabularyPtr vocabulary)
    : vocabulary_(std::move(vocabulary)),
      d_length_(vocabulary_->getDescritorSize()),
      clusters_n_(static_cast<int>(pow(vocabulary_->getBranchingFactor(),
                                       vocabulary_->getDepthLevels()))),
      v_length_(static_cast<int>(pow(vocabulary_->getBranchingFactor(),
                                     vocabulary_->getDepthLevels()) *
                                 vocabulary_->getDescritorSize() * 8)) {}

void VLAD::findCentroid(const cv::Mat &desc, cv::Mat &centroid,
                        unsigned int &id_centroid) {
  // Find the closest centroid in the vocabulary
  id_centroid = vocabulary_->transform(desc);
  vocabulary_->getWord(id_centroid).convertTo(centroid, CV_8UC1);
}

VLADVec VLAD::computeVLAD(const cv::Mat &x) {
  cv::Mat descriptor;
  cv::Mat diff, sum;
  x.convertTo(descriptor, CV_8UC1);

  // Init an empty descriptor
  cv::Mat vlad = cv::Mat::zeros(clusters_n_, d_length_, CV_8UC1);

  // Find the closest centroid and compute the VLAD
  cv::Mat closest_centroid;
  const int descriptor_cols = descriptor.cols;
  cv::Mat partial(cv::Size(descriptor_cols, 1), CV_8UC1);
  unsigned int id_centroid;

  for (int t = 0; t < descriptor.rows; t++) {
    partial = descriptor.row(t);
    findCentroid(partial, closest_centroid, id_centroid);

    closest_centroid.convertTo(closest_centroid, CV_8UC1);

    cv::bitwise_xor(partial, closest_centroid, diff);
    cv::bitwise_or(diff, vlad.row(static_cast<int>(id_centroid)),
                   vlad.row(static_cast<int>(id_centroid)));
  }

  // return the computed descriptor
  return vlad;
}

double VLAD::computeScore(const VLADVec &x, const VLADVec &y) const {
  cv::Mat res;
  cv::bitwise_xor(x, y, res);
  double norm = (v_length_ - cv::norm(res, cv::NORM_HAMMING)) / v_length_;
  BOOST_LOG_TRIVIAL(debug) << ">>>>>>>>>>>>>>>>>>>>>>>>>PR SCORE: " << norm
                           << std::endl;
  return norm;
}