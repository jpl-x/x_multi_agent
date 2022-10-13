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

#include "x/ekf/simple_state.h"

#include <utility>

using namespace x;

SimpleState::SimpleState(Vectorx dynamic_state, const Vectorx& positions_state,
                         Vectorx orientations_state, Vectorx features_state,
                         Matrix cov, std::vector<int> anchor_idxs)
    : dynamic_state_(std::move(dynamic_state)),
      positions_state_(positions_state),
      orientations_state_(std::move(orientations_state)),
      features_state_(std::move(features_state)),
      anchor_idxs_(std::move(anchor_idxs)),
      cov_(std::move(cov)),
      n_poses_(static_cast<int>(positions_state.rows()) / 3) {}

AttitudeList SimpleState::getCameraAttitudesList() {
  // Get camera orientation window from state

  // Initialize list of n attitudes
  AttitudeList attitude_list(n_poses_, x::Attitude());

  // Assign each translation
  for (int i = 0; i < n_poses_; i++) {
    const x::Attitude attitude(
        orientations_state_(4 * i), orientations_state_(4 * i + 1),
        orientations_state_(4 * i + 2), orientations_state_(4 * i + 3));
    attitude_list[i] = attitude;
  }

  return attitude_list;
}

TranslationList SimpleState::getCameraPositionsList() {
  // Initialize list of n translations
  x::TranslationList position_list(n_poses_, x::Translation());

  // Assign each translation
  for (int i = 0; i < n_poses_; i++) {
    const x::Translation translation(
        positions_state_(3 * i) + translation_[0],
        positions_state_(3 * i + 1) + translation_[1],
        positions_state_(3 * i + 2) + translation_[2]);
    position_list[i] = translation;
  }

  return position_list;
}