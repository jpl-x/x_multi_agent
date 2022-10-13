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

#if !defined(X_VIO_MULTI_SLAM_UPDATE_H_) && defined(MULTI_UAV)
#define X_VIO_MULTI_SLAM_UPDATE_H_

#include <boost/math/distributions.hpp>
#include <memory>

#include "x/ekf/ci.h"
#include "x/ekf/simple_state.h"
#include "x/ekf/state.h"
#include "x/vio/state_manager.h"
#include "x/vio/tools.h"
#include "x/vio/types.h"
#include "x/vio/update.h"

namespace x {
/**
 * Multi UAV SLAM update.
 *
 */
class MultiSlamUpdate : public Update {
 public:
  /**
   * Default constructor
   */
  MultiSlamUpdate(){};
  /**
   * @brief Construct a new Multi Slam Update object
   *
   * @param[in] trks Current SLAM tracks
   * @param[in] quats Camera quaternion states.
   * @param[in] poss Camera position states.
   * @param[in] feature_states Feature state vector.
   * @param[in] anchor_idxs Anchor pose indexes of inverse-depth SLAM
   * @param[in] cov_s Error state covariance.
   * @param[in] n_poses_max Maximum number of poses in sliding window.
   * @param[in] sigma_landmark Standard deviation of feature measurement in
   * global frame
   * @param[in] slam_matches
   * @param[in] ci_slam_w CI weight
   */
  MultiSlamUpdate(const x::TrackList& trks, const x::AttitudeList& quats,
                  const x::TranslationList& poss, const Matrix& feature_states,
                  const std::vector<int>& anchor_idxs, const Matrix& cov_s,
                  int n_poses_max, double sigma_landmark,
                  SlamMatches& slam_matches, double ci_slam_w);

  /**
   * @brief Get the Multi UAV
   *
   * @return std::vector<std::shared_ptr<Matrix>>
   */
  std::vector<std::shared_ptr<Matrix>> getMultiUAVJacobians() {
    return H_list_;
  };

  /**
   * @brief Get the Multi UAV Residuals
   *
   * @return std::vector<std::shared_ptr<Matrix>>
   */
  std::vector<std::shared_ptr<Matrix>> getMultiUAVResiduals() {
    return res_list_;
  };

  /**
   * @brief Get the Multi UAV Covariance object
   *
   * @return std::vector<std::shared_ptr<Matrix>>
   */
  std::vector<std::shared_ptr<Matrix>> getMultiUAVCovariance() {
    return P_list_;
  };

  /**
   * @brief Get the Multi UAV innovation covariance (S)
   *
   * @return std::vector<std::shared_ptr<Matrix>>
   */
  std::vector<std::shared_ptr<Matrix>> getMultiUAVS() { return S_list_; };

 private:
  // Covariance Intersection class
  CovarianceIntersection ci;

  double ci_slam_w_;

  // lists needed for the EKF update
  std::vector<std::shared_ptr<Matrix>> H_list_, S_list_, P_list_, res_list_;

  static unsigned constexpr kSlamMatchJacRows = 3;
  using SlamMatchJacBlock = Eigen::Matrix<double, kSlamMatchJacRows, kJacCols>;

  /**
   * @brief Process one SLAM-SLAM match
   *
   * @param[in] C_q_G
   * @param[in] G_p_C
   * @param[in] feature_states
   * @param[in] anchor_idx
   * @param[in] feature_id
   * @param[in] P
   * @param[in] n_poses_max
   * @param[in] other_C_q_G
   * @param[in] other_G_p_C
   * @param[in] other_feature_states
   * @param[in] other_anchor_idx
   * @param[in] other_feature_id
   * @param[in] other_P
   * @param[in] other_n_poses_max
   * @param[in] var_landmark
   * @param[in] relative_q
   * @param[in] relative_t
   */
  void processOneMatch(const x::AttitudeList& C_q_G,
                       const x::TranslationList& G_p_C,
                       const Matrix& feature_states, int anchor_idx,
                       int feature_id, const Matrix& P, int n_poses_max,
                       const AttitudeList& other_C_q_G,
                       const TranslationList& other_G_p_C,
                       const Matrix& other_feature_states, int other_anchor_idx,
                       int other_feature_id, const Matrix& other_P,
                       int other_n_poses_max, double var_landmark,
                       const Quaternion& relative_q, const Vector3& relative_t);
};
}  // namespace x

#endif  // X_VIO_SLAM_UPDATE_H_
