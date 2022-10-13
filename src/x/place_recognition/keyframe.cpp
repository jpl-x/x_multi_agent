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

#include "x/place_recognition/keyframe.h"

#include <utility>

using namespace x;

Keyframe::Keyframe(const State& state, std::vector<int>& anchors,
                   TrackList msckf_tracks, TrackList slam_tracks,
                   TrackList opp_tracks)
    : msckf_tracks_(std::move(msckf_tracks)),
      slam_tracks_(std::move(slam_tracks)),
      opp_tracks_(std::move(opp_tracks)),
      anchors_(anchors) {
  state_ = std::make_shared<SimpleState>(
      state.getDynamicStates(), state.getPositionArray(),
      state.getOrientationArray(), state.getFeatureArray(),
      state.getCovariance(), anchors);
}

cv::Mat Keyframe::getDescriptors() {
  cv::Mat descriptors;
  for (auto& i : msckf_tracks_) {
    descriptors.push_back(i.back().getDescriptor());
  }
  for (auto& i : slam_tracks_) {
    descriptors.push_back(i.back().getDescriptor());
  }
  for (auto& i : opp_tracks_) {
    descriptors.push_back(i.back().getDescriptor());
  }
  return descriptors;
}

void Keyframe::setVLAD(const VLADVec& vlad) { vlad_vec_ = vlad; }

void Keyframe::setOtherUavId(const int uav_id) { uav_ids_.insert(uav_id); }

bool Keyframe::findOtherUavId(const int uav_id) const {
  return uav_ids_.find(uav_id) != uav_ids_.end();
}
