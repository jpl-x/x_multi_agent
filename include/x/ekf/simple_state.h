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

#ifndef X_SIMPLE_STATE_H_
#define X_SIMPLE_STATE_H_

#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <vector>

#include "x/common/types.h"
#include "x/vio/types.h"
#include "x/vision/types.h"

namespace x {
class SimpleState {
 public:
  SimpleState() = delete;
  SimpleState(Vectorx dynamic_state, const Vectorx& positions_state,
              Vectorx orientations_state, Vectorx features_state, Matrix cov,
              std::vector<int> anchor_idxs);

  [[nodiscard]] int nPosesMax() const { return n_poses_; }
  [[nodiscard]] int nFeaturesMax() const {
    return static_cast<int>(features_state_.rows()) / 3;
  }
  [[nodiscard]] Vectorx getDynamicState() const { return dynamic_state_; }
  [[nodiscard]] Vectorx getPositionState() const { return positions_state_; }
  [[nodiscard]] Vectorx getOrientationState() const {
    return orientations_state_;
  }
  [[nodiscard]] Vectorx getFeatureState() const { return features_state_; }
  [[nodiscard]] Matrix getCovariance() const { return cov_; }
  [[nodiscard]] std::vector<int> getAnchorIdxs() const { return anchor_idxs_; }
  [[nodiscard]] int getErrorStateSize() const {
    return static_cast<int>(cov_.cols());
  }
  [[nodiscard]] Quaternion getRotation() const { return rotation_; }
  [[nodiscard]] Vector3 getTranslation() const { return translation_; }
  [[nodiscard]] int getAnchorIdat(int id) const { return anchor_idxs_[id]; }

  Vector3 getLastPose() { return positions_state_.tail(3); }
  [[nodiscard]] int nErrorStates() const {
    return dynamic_state_.size() - 1 + positions_state_.rows() +
           orientations_state_.rows() / 4 * 3 + features_state_.rows();
  }

  AttitudeList getCameraAttitudesList();
  TranslationList getCameraPositionsList();

 private:
  const Vectorx dynamic_state_;
  const Vectorx positions_state_;
  const Vectorx orientations_state_;
  const Vectorx features_state_;
  const Quaternion rotation_ = Quaternion(Eigen::Matrix3d::Identity());
  const Vector3 translation_ = Vector3::Zero();
  const std::vector<int> anchor_idxs_;
  const Matrix cov_;
  const int n_poses_ = -1;
};
}  // namespace x

#endif  // X_SIMPLE_STATE_H_
