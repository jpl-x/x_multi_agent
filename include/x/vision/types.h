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

#ifndef X_HAT_TYPES_H
#define X_HAT_TYPES_H

#include <x/common/types.h>
// #include <x/ekf/simple_state.h>
#include <x/vision/feature.h>
#include <x/vision/tiled_image.h>
#include <x/vision/track.h>

#include <memory>
#include <string>
#include <vector>

#define JPL_VPP_WARN_STREAM(x) \
  std::cerr << "\033[0;33m[WARN] " << x << "\033[0;0m" << std::endl;

namespace x {

using Matrix3 = Eigen::Matrix3d;
using QuaternionArray = std::vector<Quaternion>;
using Vector3Array = std::vector<Vector3>;
// Feature match
struct Match {
  Feature previous;
  Feature current;
};
// Pixel coordinates
struct PixelCoor {
  int x;  // x pixel coordinate
  int y;  // y pixel coordinate
};

typedef std::vector<Feature> FeatureList;
typedef std::shared_ptr<Feature> FeaturePtr;
typedef std::vector<FeaturePtr> FeatureListPtr;

typedef std::vector<Track> TrackList;
typedef std::shared_ptr<Track> TrackPtr;
typedef std::vector<std::shared_ptr<Track>> TrackListPtr;

typedef std::vector<Match> MatchList;
typedef std::vector<TiledImage> ImagePyramid;
typedef std::vector<cv::KeyPoint> Keypoints;
typedef std::shared_ptr<std::vector<unsigned long long>> OppIDListPtr;

typedef cv::Mat Descriptors;

typedef std::vector<float> Intensities;
struct IntensityMatches {
  Intensities prev;
  float curr;
};

struct PRfeatures {
  Keypoints keypoints = {};
  Descriptors descriptors = {};
  unsigned long long track_id = -1;
  PRfeatures(unsigned long long track_unique_id) : track_id(track_unique_id){};
  PRfeatures() = default;
  bool empty() const { return descriptors.empty() || keypoints.empty(); }
};
typedef std::vector<PRfeatures> PRList;

typedef unsigned long long uniqueId;

class SimpleState;
struct MsckfMatch {
  std::shared_ptr<SimpleState> state;
  int uav_id = -1;
  TrackPtr received_track_ptr;
  uniqueId id_current_track = -1;  // use pointer to the current track matched.
                                   // Can we ensure it's not getting destroyed?
  uniqueId id_received_track = -1;

  MsckfMatch(int uav_id, uniqueId id_current_track, uniqueId id_received_track,
             TrackPtr received_track, std::shared_ptr<SimpleState> state)
      : state(state),
        uav_id(uav_id),
        id_current_track(id_current_track),
        id_received_track(id_received_track) {
    received_track_ptr = received_track;
  };
};
using  MsckfMatches = std::vector<MsckfMatch>;

struct SlamMatch {
  std::shared_ptr<SimpleState> state;
  int uav_id = -1;
  int current_feature_id =
      -1;  // Id of the feature in the feature state vector ( in the state: )
  int received_feature_id = -1;
  SlamMatch(int uav_id, int current_feature_id, int received_feature_id,
            std::shared_ptr<SimpleState> state)
      : state(state),
        uav_id(uav_id),
        current_feature_id(current_feature_id),
        received_feature_id(received_feature_id){};
  SlamMatch() = delete;
};
typedef std::vector<SlamMatch> SlamMatches;

enum MatchType { MSCKF, SLAM };

struct MsckfSlamMatch {
  std::shared_ptr<SimpleState> state;
  int uav_id = -1;
  int feature_id = -1;
  TrackPtr other_track_ptr;
  uniqueId id_track = -1;
  MatchType type;

  MsckfSlamMatch(int uav_id, int current_feature_id, TrackPtr other_track,
                 std::shared_ptr<SimpleState> state)
      :  // if the msckf is in the other UAV
        state(state),
        uav_id(uav_id),
        feature_id(current_feature_id),
        other_track_ptr(other_track),
        type(MatchType::SLAM)  // the current UAV is slam
        {};

  MsckfSlamMatch(int uav_id, int other_feature_id, uniqueId current_track_id,
                 std::shared_ptr<SimpleState> state)
      :  // if the msckf is in the current UAV
        state(state),
        uav_id(uav_id),
        feature_id(other_feature_id),
        id_track(current_track_id),
        type(MatchType::MSCKF)  // the current UAV is msckf
        {};

  MsckfSlamMatch() = delete;

};
typedef std::vector<MsckfSlamMatch> MsckfSlamMatches;

struct OppMatch {
  std::shared_ptr<SimpleState> state;
  int uav_id = -1;
  uniqueId id_current_track = -1;
  uniqueId id_received_track = -1;
  int received_feature_id = -1;
  TrackPtr received_track_ptr;
  MatchType type;
  OppMatch(int uav_id, uniqueId id_current_track, uniqueId id_received_track,
           int received_feature_id, TrackPtr received_track,
           std::shared_ptr<SimpleState> state, MatchType type)
      : state(state),
        uav_id(uav_id),
        id_current_track(id_current_track),
        id_received_track(id_received_track),
        received_feature_id(received_feature_id),
        type(type) {
    received_track_ptr = received_track;
  };
  OppMatch() = delete;
};
typedef std::vector<OppMatch> OppMatches;

// TODO(jeff) Get rid of this. Use Eigen::Quaternion instead.
struct Attitude {
  double ax = 0;  //< quat x
  double ay = 0;  //< quat y
  double az = 0;  //< quat z
  double aw = 0;  //< quat orientation
  Attitude(double ax, double ay, double az, double aw)
      : ax(ax), ay(ay), az(az), aw(aw) {}
  Attitude() = default;
};

// TODO(jeff) Get rid of this. Use Vector3 instead.
struct Translation {
  double tx = 0;  //< cam position x
  double ty = 0;  //< cam position y
  double tz = 0;  //< cam position z
  Translation(double tx, double ty, double tz) : tx(tx), ty(ty), tz(tz) {}
  Translation(void) = default;
};

// TODO(jeff) Get rid of below and use the above
using AttitudeList = std::vector<Attitude>;
using TranslationList = std::vector<Translation>;

}  // namespace x

#endif
