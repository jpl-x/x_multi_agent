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

#ifndef X_VIO_VIO_UPDATER_H_
#define X_VIO_VIO_UPDATER_H_

#include <Eigen/QR>
#include <memory>

#include "x/ekf/simple_state.h"
#include "x/ekf/state.h"
#include "x/ekf/updater.h"
#include "x/vio/msckf_slam_update.h"
#include "x/vio/multi_slam_update.h"
#include "x/vio/slam_update.h"
#include "x/vio/state_manager.h"
#include "x/vio/tools.h"
#include "x/vio/track_manager.h"
#include "x/vio/types.h"

namespace x {
class VioUpdater : public Updater {
 public:
  /**
   * @brief Default constructor.
   */
  VioUpdater() = default;

  /**
   * @brief Construct with known parameters.
   */
  VioUpdater(Tracker &tracker, StateManager &state_manager,
             TrackManager &track_manager, double sigma_img, double sigma_range,
             double rho_0, double sigma_rho_0, int min_track_length,
             double sigma_landmark = 0, double ci_msckf_w = -1.0,
             double ci_slam_w = -1.0, int iekf_iter = 1);

  /**
   * @brief Set the input measurements for the independent VIO system
   * @param measurement
   */
  void setMeasurement(const VioMeasurement &measurement);

  /**
   * @brief Get measurement timestamp.
   */
  [[nodiscard]] double getTime() const override;

  /**
   * @brief Get reference to the tracker debug image.
   */
  [[nodiscard]] TiledImage &getMatchImage();

  /**
   * @brief Get reference to the track manager debug image.
   */
  [[nodiscard]] TiledImage &getFeatureImage();


#ifdef MULTI_UAV
  /**
   * @brief Get the Msckf Tracks
   *
   * @param[out] tracks
   */
  void getMsckfTracks(TrackList &tracks);

  /**
   * @brief Get the Opp Tracks
   *
   * @param tracks
   */
  void getOppTracks(TrackList &tracks);

  /**
   * @brief Get the Slam Tracks and what is needed by the slam update
   *
   * @param[out] tracks
   * @param[out] anchor_idxs
   * @param[out] n_poses_max
   */
  void getSlamTracks(TrackList &tracks, std::vector<int> &anchor_idxs,
                     int n_poses_max);

  /**
   * @brief Get the Slam Tracks
   *
   * @param[out] tracks
   */
  void getSlamTracks(TrackList &tracks);
#endif

  /**
   * @brief Get list of MSCKF inliers' 3D coordinates.
   */
  [[nodiscard]] Vector3dArray getMsckfInliers() const;

  /**
   * @brief Get list of MSCKF outliers' 3D coordinates.
   */
  [[nodiscard]] Vector3dArray getMsckfOutliers() const;

 private:
  // VIO measurement: image + optional range and sun angle.
  VioMeasurement measurement_;

  Tracker tracker_;
  StateManager state_manager_;
  TrackManager track_manager_;

  // Tracker debug image.
  TiledImage match_img_;

  // Track manager debug image.
  TiledImage feature_img_;

  // Standard deviation of feature measurement [in normalized coordinates].
  double sigma_img_{};

  // Standard deviation of landmarks measurement.
  double sigma_landmark_{};

  // Standard deviation of range measurement noise [m].
  double sigma_range_{};

  // Initial inverse depth of SLAM features [1/m].
  double rho_0_{};

  // Initial standard deviation of SLAM inverse depth [1/m].
  double sigma_rho_0_{};

  // Minimum track length for a visual feature to be processed.
  int min_track_length_{};

  // Covariance intersection weights. Used only by the MULTI UAV system.
  double ci_msckf_w_{}, ci_slam_w_{};

  // All track members assume normalized undistorted feature coordinates.
  TrackList msckf_trks_;  // Normalized tracks for MSCKF update
  TrackList msckf_short_trks_;
  // New SLAM features initialized with semi-infinite depth uncertainty
  // (standard SLAM)
  TrackList new_slam_std_trks_;
  // New SLAM features initialized with MSCKF tracks (MSCKF-SLAM)
  TrackList new_msckf_slam_trks_;
  TrackList slam_trks_;

#ifdef MULTI_UAV
  // Opportunistic tracks. These can became SLAM or MSCK tracks.
  TrackList opt_trks_;
  double thr_comm_distance_ =
      15.0;  // max distance to perform the Place Recognition

  int n_poses_max_ = 0;
  Vector3 last_pose_ = Vector3(0, 0, 0);  // Last pose of the current UAV, neede
#endif

#if defined(MULTI_UAV) && defined(REQUEST_COMM)
  int frames_min_distance_ = 0;
  // to compute the distance.
  Vector3 prev_diff_ = Vector3(0, 0, 0);

#endif
  std::vector<unsigned int> lost_slam_trk_idxs_;
  // 3D coordinates prior for MSCKF feature inliers.
  Vector3dArray
      msckf_inliers_;  // 3D coordinates prior for MSCKF feature inliers.
  Vector3dArray
      msckf_outliers_;  // 3D coordinates prior for MSCKF feature outliers.

  // SLAM update object.
  SlamUpdate slam_;

  // MSCKF-SLAM update object
  MsckfSlamUpdate msckf_slam_;

  /**
   * @brief Measurement processing
   *
   * Only happens once, and is stored if measurement later re-applied. Here this
   * where image processing takes place.
   *
   * @param[in] state State prior
   */
  void preProcess(const State &state) override;

  /**
   * @brief Stuff that needs happen before the Kalman update. This function will
   * NOT be called at  each IEKF iteration. Here, this corresponds to state
   * management.
   *
   * @param[in,out] state Update state
   * @return True if an update should be constructed
   */
  [[nodiscard]] bool preUpdate(State &state) override;

  /**
   * @brief Stuff that needs happen before the MSCKF update with short tracks.
   *
   * @return True if an update should be constructed
   */
  [[nodiscard]] bool preUpdateShortMsckf() override;

#ifdef MULTI_UAV

  /**
   * @brief Check if there are slam matches.
   *
   * @return true if the CI update should be constructed
   */
  [[nodiscard]] bool preUpdateCI() override;

  /**
   * @brief construct MSCKF update for shor tracks
   *
   * @param state
   * @param h
   * @param res
   * @param r
   * @param S_list
   * @param P_list
   * @param H_list
   * @param res_list
   */
  void constructShortMsckfUpdate(
      const State &state, Matrix &h, Matrix &res, Matrix &r,
      std::vector<std::shared_ptr<Matrix>> &S_list,
      std::vector<std::shared_ptr<Matrix>> &P_list,
      std::vector<std::shared_ptr<Matrix>> &H_list,
      std::vector<std::shared_ptr<Matrix>> &res_list) override;

  /**
   * @brief Construct the Jacobians, the residuals, the Innovation covariance
   * and the covariance using the CI algorithm
   *
   * @param[in] state
   * @param[out] S_list
   * @param[out] P_list
   * @param[out] H_list
   * @param[out] res_list
   */
  void constructSlamCIUpdate(
      const State &state, std::vector<std::shared_ptr<Matrix>> &S_list,
      std::vector<std::shared_ptr<Matrix>> &P_list,
      std::vector<std::shared_ptr<Matrix>> &H_list,
      std::vector<std::shared_ptr<Matrix>> &res_list) override;

  /**
   * @brief Construct the Jacobians, the residuals and the covariance
   * If there are MSCKF matches in a MULTI_UAV setup it performs a CI returning
   * the lists of the Jacobians, the residuals, the Innovation covariance and
   * the covariance using the CI algorithm for the MSCKF CI-EKF update.
   *
   * @param[in] state
   * @param[out] h
   * @param[out] res
   * @param[out] r
   * @param[out] S_list
   * @param[out] P_list
   * @param[out] H_list
   * @param[out] res_list
   */
  void constructUpdate(const State &state, Matrix &h, Matrix &res, Matrix &r,
                       std::vector<std::shared_ptr<Matrix>> &S_list,
                       std::vector<std::shared_ptr<Matrix>> &P_list,
                       std::vector<std::shared_ptr<Matrix>> &H_list,
                       std::vector<std::shared_ptr<Matrix>> &res_list) override;

  MsckfMatches msckf_matches_;
  SlamMatches slam_matches_;
#else
  /**
   * @brief Update construction.
   *
   * Prepares measurement Jacobians, residual and noice matrices and applies
   * (iterated) EKF update. This function WILL be called at each IEKF iteration.
   *
   * @param[in] state Update state
   * @param[out] h Measurement Jacobian matrix
   * @param[out] res Measurement residual vector
   * @param[out] r Measurement noise covariance matrix
   */
  void constructUpdate(const State &state, Matrix &h, Matrix &res,
                       Matrix &r) override;

  /**
   * @brief construct MSCKF update for shor tracks
   *
   * @param state
   * @param h
   * @param res
   * @param r
   */
  void constructShortMsckfUpdate(const State &state, Matrix &h, Matrix &res,
                                 Matrix &r) override;
#endif

  /**
   * @brief Post-update work.
   *
   * Stuff that need to happen after the Kalman update. Here this SLAM
   * feature initialization.
   *
   * @param[in,out] state Update state.
   * @param[in] correction Kalman state correction.
   */
  void postUpdate(State &state, const Matrix &correction) override;

  /** @brief QR decomposition
   *
   *  Computes the QR decomposition of the MSCKF update
   *  Jacobian and updates the other terms according to Mourikis'
   *  2007 paper.
   *
   *  @param[in,out] h Jacobian matrix of MSCKF measurements
   *  @param[in,out] res MSCKF measurement residuals
   *  @param[in,out] R Measurement noise matrix
   */
  void applyQRDecomposition(Matrix &h, Matrix &res, Matrix &R) const;

  friend class VIO;
};
}  // namespace x

#endif  // XVIO_MEASUREMENT_H_
