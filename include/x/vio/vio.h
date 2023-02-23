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

#ifndef X_VIO_VIO_H_
#define X_VIO_VIO_H_

#include <thread>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <mutex>

#include "x/common/types.h"
#include "x/ekf/ekf.h"
#include "x/ekf/simple_state.h"
#include "x/ekf/state.h"
#include "x/vio/state_manager.h"
#include "x/vio/track_manager.h"
#include "x/vio/types.h"
#include "x/vio/vio_updater.h"
#include "x/vision/camera.h"
#include "x/vision/tiled_image.h"
#include "x/vision/tracker.h"

#ifdef MULTI_UAV
#include "x/place_recognition/keyframe.h"
#include "x/place_recognition/place_recognition.h"
#endif

namespace x {
namespace fsm = std::filesystem;
class VIO {
 public:
  VIO();

  /**
   * Return if the ekf is initialized
   *
   * @return bool
   */
  [[nodiscard]] bool isInitialized() const;

  /**
   * Init the ekf with a certain timestamp time
   *
   * @param time
   */
  void initAtTime(const double &time);

  /**
   * Setup VIO parameters
   *
   * @param params
   */
  void setUp(const Params &params);

  /**
   * Load new range measurements
   *
   * @param range_measurement
   */
  void setLastRangeMeasurement(const RangeMeasurement &range_measurement);

  /**
   * Load new sun angle measurements
   *
   * @param angle_measurement
   */
  void setLastSunAngleMeasurement(const SunAngleMeasurement &angle_measurement);

  /**
   * Pass IMU measurements to EKF for propagation.
   *
   * @param[in] timestamp
   * @param[in] msg_seq Message ID.
   * @param[in] w_m Angular velocity (gyroscope).
   * @param[in] a_m Specific force (accelerometer).
   * @return The propagated state.
   */
  std::optional<State> processImu(const double &timestamp, unsigned int seq,
                                  const Vector3 &w_m, const Vector3 &a_m);

  /**
   * Creates an update measurement from image and pass it to EKF.
   *
   * @param[in] timestamp Image timestamp.
   * @param[in] seq Image sequence ID.
   * @param[in,out] match_img Image input, overwritten as tracker debug image
   *                          in output.
   * @param[out] feature_img Track manager image output.
   * @return The updated state, or invalide if the update could not happen.
   */
  std::optional<State> processImageMeasurement(const double &timestamp,
                                               unsigned int seq,
                                               TiledImage &match_img,
                                               TiledImage &feature_img);

  /**
   * Creates an update measurement from visual matches and pass it to EKF.
   *
   * @param[in] timestamp Image timestamp.
   * @param[in] seq Image sequence ID.
   * @param[in,out] match_vector Feature matches vector.
   * @param[out] match_img Tracker image output.
   * @param[out] feature_img Track manager image output.
   * @return The updated state, or invalid if the update could not happen.
   */
  std::optional<State> processMatchesMeasurement(
      const double &timestamp, unsigned int seq,
      const std::vector<double> &match_vector, TiledImage &match_img,
      TiledImage &feature_img);

  /**
   * Compute cartesian coordinates of SLAM features for input state.
   *
   * @param[in] state Input state.
   * @return A vector with the 3D cartesian coordinates.
   */
  std::vector<Vector3> computeSLAMCartesianFeaturesForState(const State &state);

  /**
   * Get MSCKF inliers and outliers
   *
   * @param inliers
   * @param outliers
   */
  void getMsckfFeatures(Vector3dArray &inliers, Vector3dArray &outliers);

  /**
   * Load parameters from a yaml file into a Params struct
   *
   * @param path, path to the yaml file
   * @return Params
   */
  Params loadParamsFromYaml(fsm::path &path);
#ifdef MULTI_UAV

#ifndef REQUEST_COMM

  /**
   * @brief get the the data that need to be sent to the other UAVs
   *
   * @param[out] msckf_tracks
   * @param[out] slam_tracks
   * @param[out] anchor_idxs
   * @param[out] n_poses_max
   * @param[out] opp_tracks
   */
  void getDataToSend(std::shared_ptr<SimpleState> &state_ptr,
                     const State &state, TrackList &msckf_tracks,
                     TrackList &slam_tracks, std::vector<int> &anchor_idxs,
                     TrackList &opp_tracks);

#endif

  /**
   * @brief check whether the current UAV knows about places the other UAV is
   * visiting.
   *
   * @param[in] uav_id
   * @param[in] descriptors
   * @param[out] state
   * @param[out] msckf_tracks
   * @param[out] slam_tracks
   * @param[out] anchor_idxs
   * @param[out] opp_features
   */
  void processOtherRequests(int uav_id, cv::Mat &descriptors,
                            std::shared_ptr<SimpleState> &state,
                            TrackList &msckf_tracks, TrackList &slam_tracks,
                            std::vector<int> &anchor_idxs,
                            TrackList &opp_tracks);

  /**
   * Get last frame descriptors
   *
   * @return cv::Mat containing descriptors
   */
  cv::Mat getDescriptors();

  /**
   * @brief Process the data received by the other UAVs
   *
   * @param[in] uav_id id of the sender
   * @param[in] dynamic_state
   * @param[in] positions_state
   * @param[in] orientations_state
   * @param[in] features_state
   * @param[in] cov
   * @param[in] received_msckf_trcks
   * @param[in] received_slam_trcks
   * @param[in] received_opp_tracks
   * @param[in] anchor_idxs
   * @param[in,out] dst image for showing the features correspondences
   */
  std::optional<State> processOtherMeasurements(
      double timestamp, int uav_id, const Vectorx &dynamic_state,
      const Vectorx &positions_state, const Vectorx &orientations_state,
      const Vectorx &features_state, const Matrix &cov,
      const TrackListPtr &received_msckf_trcks_ptr,
      const TrackListPtr &received_slam_trcks_ptr,
      const TrackListPtr &received_opp_tracks_ptr,
      const std::vector<int> &anchor_idxs, cv::Mat &dst);

#endif

 private:
  /**
   * Extended Kalman filter estimation back end.
   */
  Ekf ekf_;

  /**
   * VIO EKF updater.
   *
   * Constructs and applies an EKF update from a VIO measurement. The EKF
   * class owns a reference to this object through Updater abstract class,
   * which it calls to apply the update.
   */
  VioUpdater vio_updater_;

  Params params_;

  /**
   * Minimum baseline for MSCKF (in normalized plane).
   */
  double msckf_baseline_x_n_, msckf_baseline_y_n_;

  Camera camera_;
  Tracker tracker_;
  TrackManager track_manager_;
  StateManager state_manager_;
  RangeMeasurement last_range_measurement_;
  SunAngleMeasurement last_angle_measurement_;
  bool initialized_{false};

  /**
   * Import a feature match list from a std::vector.
   *
   * @param[in] match_vector Input vector of matches.
   * @param[in] seq Image sequence ID.
   * @param[out] img_plot Debug image.
   * @return The list of matches.
   */
  MatchList importMatches(const std::vector<double> &match_vector,
                          unsigned int seq, x::TiledImage &img_plot) const;

  std::vector<Vector3> imu_data_batch_{};
  bool initialize_start_{false};
  bool self_init_start_{false};

#ifdef MULTI_THREAD
  TiledImage matches_img_;  // current UAV image to store the matches

  std::mutex mtx_;
#endif
#ifdef MULTI_UAV
  Keyframe candidate_keyframe_;
  std::shared_ptr<PlaceRecognition>
      place_recognition_;  // this can be nullptr when MULTI_UAV is off
#endif
};
}  // namespace x

#endif
