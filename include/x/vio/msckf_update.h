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

#ifndef X_VIO_MSCKF_UPDATE_H_
#define X_VIO_MSCKF_UPDATE_H_

#include <memory>

#include "x/ekf/ci.h"
#include "x/ekf/simple_state.h"
#include "x/vio/state_manager.h"
#include "x/vio/types.h"
#include "x/vio/update.h"
#include "x/vision/triangulation.h"

namespace x {
/**
 * MSCKF update.
 *
 * Implementation of the Multi-State Contraint Kalman Filter.
 * (MSCKF). As presented Mourikis 2007 ICRA paper.
 */
class MsckfUpdate : public Update {
 public:
  /**
   * Constructor.
   *
   * Does the full update matrix construction job.
   *
   * @param[in] tkrs Feature tracks in normalized coordinates
   * @param[in] quats Camera quaternion states
   * @param[in] poss Camera position states
   * @param[in] triangulator Feature triangulator
   * @param[in] cov_s Error state covariance
   * @param[in] n_poses_max Maximum number of poses in sliding window.
   * @param[in] sigma_img Standard deviation of feature measurement
   *                      [in normalized coordinates]
   */
  MsckfUpdate(const x::TrackList &trks, const x::AttitudeList &quats,
              const x::TranslationList &poss, const Matrix &cov_s,
              int n_poses_max, double sigma_img
#ifdef MULTI_UAV
              ,
              MsckfMatches &tracks_matches, double ci_msckf_w
#endif
  );

  /***************************** Getters **********************************/

  /**
   * Returns a constant reference to the inliers 3D cartesian coodinates.
   */
  [[nodiscard]] const Vector3dArray &getInliers() const { return inliers_; };

  /**
   * Returns a constant reference to the outliers 3D cartesian coodinates.
   */
  [[nodiscard]] const Vector3dArray &getOutliers() const { return outliers_; };

#ifdef MULTI_UAV
  /**
   * Get all the elements needed for the multi UAV MSCKF update
   */
  std::vector<std::shared_ptr<Matrix>> getMultiUAVJacobians() {
    return H_list_;
  };
  std::vector<std::shared_ptr<Matrix>> getMultiUAVResiduals() {
    return res_list_;
  };
  std::vector<std::shared_ptr<Matrix>> getMultiUAVCovariance() {
    return P_list_;
  };
  std::vector<std::shared_ptr<Matrix>> getMultiUAVS() { return S_list_; };
#endif

 private:
  // TODO(jeff) set from params
  // Set up triangulation for MSCKF
  // Gauss-Newton max number of iterations
  const unsigned int max_iter_ = 10;

  // Gauss-Newton termination criterion
  const double term_ = 0.00001;

  // 3D cartesian coordinates prior for MSCKF feature inliers
  Vector3dArray inliers_;

  // 3D cartesian coordinates prior for MSCKF feature outliers
  Vector3dArray outliers_;

  /**
   * Process one feature track.
   *
   * @param[in] track Feature track in normalized coordinates
   * @param[in] C_q_G Camera attitude state list
   * @param[in] G_p_C Camera position state list
   * @param[in] P State covariance
   * @param[in] n_poses_max Maximum number of poses in sliding window.
   * @param[in] var_img Variance of feature measurement.
   *                    [in normalized coordinates].
   * @param[in] j Index of current track in the MSCKF track list
   * @param[out] row_h Current rows in stacked Jacobian
   */
  void preProcessOneTrack(const x::Track &track, const x::AttitudeList &C_q_G,
                          const x::TranslationList &G_p_C,
#ifdef MULTI_UAV
                          MsckfMatches &tracks_matches,
#endif
                          const Matrix &P, int n_poses_max, double var_img,
                          const int &j, int &row_h);

  // Problem: the feature can be triangulated after we have all the positions
  // and observations from the matched tracks. After that we have an
  // estimation the feature position in the space. The feature is then used in
  // the MSCKF update.
  // For avoiding iterating again over the tracks for finding
  // correspondences, keep track of the matches, the covariances, the states
  // and the position of all matches

  /**
   * @brief Get the Global Feature Position object
   *
   * @param[in] inverse_depth
   * @param[in] C_q_G
   * @param[in] G_p_C
   * @param[out] global_feature_pose
   */
  static void getGlobalFeaturePosition(const Vector3 &inverse_depth,
                                       const Attitude &C_q_G,
                                       const Translation &G_p_C,
                                       Vector3 &global_feature_pose);

  /**
   * Compute partial Jacobians and residuals
   *
   * @param track
   * @param P
   * @param C_q_G
   * @param G_p_C
   * @param n_poses_max
   * @param var_img
   * @param G_p_fj
   * @param multi_row_h
   * @param multi_col_h
   * @param row_h
   * @param is_multi_msckf
   */
  void processOneTrack(const x::Track &track, const Matrix &P,
                       const x::AttitudeList &C_q_G,
                       const x::TranslationList &G_p_C, int n_poses_max,
                       double var_img, const Vector3 &G_p_fj, int &multi_row_h,
                       int &multi_col_h, int &row_h,
                       bool is_multi_msckf = false);

  static void nullSpaceProjection(Matrix &H_j, Matrix &Hf_j, Matrix &res);

  // Covariance intersection weights. Used only by the MULTI UAV system
  double ci_msckf_w_;
#ifdef MULTI_UAV
  CovarianceIntersection cov_intersection_;

  // this is false when the track the MSCKF match refers to has been
  // rejected for the normal MSCKF update.
  bool proceed_with_multi_ = false;
#endif

  std::vector<std::shared_ptr<Matrix>> S_list_, P_list_, H_list_, res_list_;

  // Residual vector referring to the feature
  Matrix res_pf_;

  // Jacobian matrix of the feature and of the state referring to the feature
  Matrix jac_x_pf_, jac_pf_;
};

}  // namespace x

#endif  // X_VIO_MSCKF_UPDATE_H_
