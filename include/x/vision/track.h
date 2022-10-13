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

#ifndef X_VISION_TRACK_H_
#define X_VISION_TRACK_H_

#include <vector>

#include "x/vision/feature.h"

namespace x {

/**
 * A feature track class.
 *
 * This class is a vector of features (i.e. 2D image coordinates) ordered
 * chronologically.
 */
class Track : public std::vector<Feature> {
 public:
  /**
   * Default constructor.
   */
  Track() : std::vector<Feature>() {
    counter_++;
    id_ = counter_;
  };

  /**
   * Return the unique id of the track
   * @return id
   */
  [[nodiscard]] unsigned long long getId() const { return id_; }

  /**
   * Constructs a Track object of a given size, filled a with a given
   * feature object. Used to import tracks of other UAVs.
   *
   * @param[in] count Track size.
   * @param[in] feature Feature to initialize each track entry at.
   */
#ifdef MULTI_UAV
  Track(const size_type count, const Feature& feature, unsigned long long id)
      : std::vector<Feature>(count, feature),
        id_(id),
        descriptor_(feature.getDescriptor()){};
#else

  Track(const size_type count, const Feature& feature, unsigned long long id)
      : std::vector<Feature>(count, feature),
        id_(id){};
#endif
#ifdef MULTI_UAV
  /**
   * Constructs a Track object with a specified id. Used to import tracks of
   * other UAVs.
   * @param id
   */
  explicit Track(unsigned long long id) : id_(id){};

  [[nodiscard]] cv::Mat getDescriptor() const { return descriptor_; };
  void setDescriptor(const cv::Mat& descriptor) { descriptor_ = descriptor; };
#endif

 private:
  unsigned long long id_{0};
  static unsigned long long counter_;

#ifdef MULTI_UAV
  cv::Mat descriptor_;
#endif
};

}  // namespace x

#endif  // X_VISION_TRACK_H_
