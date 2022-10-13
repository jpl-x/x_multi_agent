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

#if !defined(X_KEYFRAME_H) && defined(MULTI_UAV)
#define X_KEYFRAME_H

#include <memory>
#include <set>

#include "x/ekf/simple_state.h"
#include "x/ekf/state.h"
#include "x/place_recognition/types.h"
#include "x/place_recognition/vlad.h"
#include "x/vision/feature.h"
#include "x/vision/track.h"
#include "x/vision/types.h"

namespace x {
class Keyframe {
 public:
  Keyframe() = default;

  /**
   * Keyframe constructor and keyframe parameters initialization
   *
   * @param state
   * @param anchors
   * @param msckf_tracks
   * @param slam_tracks
   * @param opp_tracks
   */
  Keyframe(const State &state, std::vector<int> &anchors,
           TrackList msckf_tracks, TrackList slam_tracks,
           TrackList opp_tracks);

  /**
   * Get the VLAD descriptor for the current keyframe
   *
   * @return VLADVec
   */
  [[nodiscard]] VLADVec getVLAD() const { return vlad_vec_; };

  /**
   * Get the State pointer for the current keyframe
   *
   * @return std::shared_ptr<SimpleState>
   */
  [[nodiscard]] std::shared_ptr<SimpleState> getState() const {
    return state_;
  };

  /**
   * Get the MSCKF tracks for the current keyframe
   *
   * @return TrackList
   */
  [[nodiscard]] TrackList getMsckfTracks() const { return msckf_tracks_; };

  /**
   * Get the SLAM tracks for the current keyframe
   *
   * @return TrackList
   */
  [[nodiscard]] TrackList getSlamTracks() const { return slam_tracks_; };

  /**
   * Get the Opportunistic tracks for the current keyframe
   *
   * @return TrackList
   */
  [[nodiscard]] TrackList getOppTracks() const { return opp_tracks_; };

  /**
   * Get the anchor ids for the current keyframe
   *
   * @return std::vector<int>
   */
  [[nodiscard]] std::vector<int> getAnchors() const { return anchors_; };

  /**
   * Set the VLAD descriptor for the current VLAD
   *
   * @param vlad
   */
  void setVLAD(const VLADVec &vlad);

  /**
   * Get the feature descriptors for the current keyframe
   *
   * @return cv::Mat
   */
  cv::Mat getDescriptors();

  /**
   * UAV ids that have already used the current keyframe
   * @param uav_id
   */
  void setOtherUavId(int uav_id);

  /**
   * Find if the UAV with uav_id has already used this keyframe
   *
   * @param uav_id
   * @return bool
   */
  [[nodiscard]] bool findOtherUavId(int uav_id) const;

 private:
  // params of the keyframe
  VLADVec vlad_vec_;
  std::shared_ptr<SimpleState> state_;
  TrackList msckf_tracks_;
  TrackList slam_tracks_;
  TrackList opp_tracks_;
  std::vector<int> anchors_;

  std::set<int> uav_ids_;  // store the UAVs that have this frame in common.
};
}  // namespace x

#endif