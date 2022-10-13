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

#include "x/vision/feature.h"

using namespace x;

Feature::Feature() = default;

Feature::Feature(const double& timestamp, double x, double y, double intensity)
    : timestamp_(timestamp), x_(x), y_(y), intensity_(intensity) {}

Feature::Feature(const double& timestamp, unsigned int frame_number, double x,
                 double y, double x_dist, double y_dist, double intensity)
    : timestamp_(timestamp),
      frame_number_(frame_number),
      x_(x),
      y_(y),
      x_dist_(x_dist),
      y_dist_(y_dist),
      intensity_(intensity) {}

Feature::Feature(const double& timestamp, unsigned int frame_number,
                 double x_dist, double y_dist, unsigned int pyramid_level,
                 float fast_score, double intensity)
    : timestamp_(timestamp),
      frame_number_(frame_number),
      x_dist_(x_dist),
      y_dist_(y_dist),
      pyramid_level_(pyramid_level),
      fast_score_(fast_score),
      intensity_(intensity) {}

bool Feature::operator==(const Feature& other) {
  return nearlyEqual(x_, other.getX()) && nearlyEqual(y_, other.getY());
}

bool Feature::nearlyEqual(double a, double b) {
  double absA = std::abs(a);
  double absB = std::abs(b);
  double diff = std::abs(a - b);

  if (a == b) {  // shortcut, handles infinities
    return true;
  } else if (a == 0 || b == 0 ||
             (absA + absB < std::numeric_limits<double>::min())) {
    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    return diff < (std::numeric_limits<double>::epsilon() *
                   std::numeric_limits<double>::min());
  } else {  // use relative error
    return diff / std::min((absA + absB), std::numeric_limits<double>::max()) <
           std::numeric_limits<double>::epsilon();
  }
}