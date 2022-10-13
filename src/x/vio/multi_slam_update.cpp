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

#include "x/vio/multi_slam_update.h"

using namespace x;

MultiSlamUpdate::MultiSlamUpdate(
    const x::TrackList &trks, const x::AttitudeList &quats,
    const x::TranslationList &poss, const Matrix &feature_states,
    const std::vector<int> &anchor_idxs, const Matrix &cov_s,
    const int n_poses_max, const double sigma_landmark,
    SlamMatches &slam_matches, const double ci_slam_w)
    : ci_slam_w_(ci_slam_w) {
  // Number of SLAM-SLAM matches
  const size_t n_matches = slam_matches.size();

  // Initialize Kalman update matrices
  /*
  res = p_w_c - p_w_r = 0
  x = x_cur-x_rec
  y = y_cur-y_rec
  z = z_cur-z_rec
  */
  const auto rows = 3 * n_matches;
  const auto cols = cov_s.cols();
  jac_ = Matrix::Zero(rows, cols);
  cov_m_diag_ = Vectorx::Ones(rows);
  res_ = Matrix::Zero(rows, 1);

  // For each track, compute residual, Jacobian and covariance block
  const double var_landmark = sigma_landmark * sigma_landmark;
  for (SlamMatch match : slam_matches) {
    std::shared_ptr<SimpleState> other_state = match.state;

    processOneMatch(
        quats, poss, feature_states, anchor_idxs[match.current_feature_id],
        match.current_feature_id, cov_s, n_poses_max,
        other_state->getCameraAttitudesList(),
        other_state->getCameraPositionsList(), other_state->getFeatureState(),
        other_state->getAnchorIdat(match.received_feature_id),
        match.received_feature_id, other_state->getCovariance(),
        other_state->nPosesMax(), var_landmark, other_state->getRotation(),
        other_state->getTranslation());
  }
}

void MultiSlamUpdate::processOneMatch(
    const x::AttitudeList &C_q_G, const x::TranslationList &G_p_C,
    const Matrix &feature_states, const int anchor_idx, const int feature_id,
    const Matrix &P, const int n_poses_max, const AttitudeList &other_C_q_G,
    const TranslationList &other_G_p_C, const Matrix &other_feature_states,
    const int other_anchor_idx, const int other_feature_id,
    const Matrix &other_P, const int other_n_poses_max,
    const double var_landmark, const Quaternion &relative_q,
    const Vector3 &relative_t) {
  const int cols = P.cols();
  const int other_cols = other_P.cols();

  Matrix h_j(Matrix::Zero(3, cols));
  Matrix other_h_j(Matrix::Zero(3, other_cols));
  Matrix res_j(Matrix::Zero(3, 1));

  // Compute current params
  // current invers depth param
  const double alpha = feature_states(feature_id * 3);
  const double beta = feature_states(feature_id * 3 + 1);
  const double rho = feature_states(feature_id * 3 + 2);

  if (anchor_idx < 0) {
    throw std::runtime_error("anchor_idx < 0");
  }
  if (rho == 0) {
    throw std::runtime_error("rho = 0");
  }

  // Current anchor pose
  Quaternion Ca_q_G;
  Ca_q_G.x() = C_q_G[anchor_idx].ax;
  Ca_q_G.y() = C_q_G[anchor_idx].ay;
  Ca_q_G.z() = C_q_G[anchor_idx].az;
  Ca_q_G.w() = C_q_G[anchor_idx].aw;

  Vector3 G_p_Ca(G_p_C[anchor_idx].tx, G_p_C[anchor_idx].ty,
                 G_p_C[anchor_idx].tz);

  // Coordinate of feature current UAV in global frame
  Vector3 G_p_fj = (1 / rho) * Ca_q_G.normalized().toRotationMatrix() *
                       Vector3(alpha, beta, 1) +
                   G_p_Ca;

  // Compute other params
  // other inverse depth param
  const double other_alpha = other_feature_states(other_feature_id * 3);
  const double other_beta = other_feature_states(other_feature_id * 3 + 1);
  const double other_rho = other_feature_states(other_feature_id * 3 + 2);

  // Other anchor pose
  Quaternion other_Ca_q_G;
  other_Ca_q_G.x() = other_C_q_G[other_anchor_idx].ax;
  other_Ca_q_G.y() = other_C_q_G[other_anchor_idx].ay;
  other_Ca_q_G.z() = other_C_q_G[other_anchor_idx].az;
  other_Ca_q_G.w() = other_C_q_G[other_anchor_idx].aw;

  Vector3 other_G_p_Ca(other_G_p_C[other_anchor_idx].tx,
                       other_G_p_C[other_anchor_idx].ty,
                       other_G_p_C[other_anchor_idx].tz);

  // Coordinate of feature other UAV in its global frame (assuming the frame is
  // different)
  Vector3 other_G_p_fj = (1 / other_rho) *
                             other_Ca_q_G.normalized().toRotationMatrix() *
                             Vector3(other_alpha, other_beta, 1) +
                         other_G_p_Ca;

  // Computing residual according to eq. 27 (Distributed Visual-Inertial
  // Cooperative Localization) assuming the two UAVs refer to different frames
  res_j = -G_p_fj + other_G_p_fj;  // res_j = G_p_fj - other_G_p_fj; res_j =
                                   // -res_j; r = -h(x,m)

  // =========================================
  //  Measurement Jacobian matrix Current state
  // =========================================
  // Set Jacobian of pose for i'th measurement of feature j (eq.22, 23)
  // Anchor position
  SlamMatchJacBlock J_anchor_pos = Matrix::Identity(3, 3);

  // Anchor attitude
  SlamMatchJacBlock J_anchor_att = -(1 / rho) *
                                   Ca_q_G.normalized().toRotationMatrix() *
                                   x::Skew(alpha, beta, 1).matrix;

  // Inverse-depth feature coordinates
  Matrix mat(Matrix::Identity(3, 3));
  mat(0, 2) = -alpha / rho;
  mat(1, 2) = -beta / rho;
  mat(2, 2) = -1 / rho;

  SlamMatchJacBlock Hf_j1 =
      (1 / rho) * Ca_q_G.normalized().toRotationMatrix() * mat;

  // =========================================
  //  Measurement Jacobian matrix Other state
  // =========================================

  // Set Jacobian of pose for i'th measurement of feature j (eq.22, 23)
  // Anchor position
  SlamMatchJacBlock other_J_anchor_pos = Matrix::Identity(3, 3);

  // Anchor attitude
  SlamMatchJacBlock other_J_anchor_att =
      -(1 / other_rho) * other_Ca_q_G.normalized().toRotationMatrix() *
      Skew(other_alpha, other_beta, 1).matrix;

  // Inverse-depth feature coordinates
  Matrix other_mat = Matrix::Identity(3, 3);
  other_mat(0, 2) = -other_alpha / other_rho;
  other_mat(1, 2) = -other_beta / other_rho;
  other_mat(2, 2) = -1 / other_rho;

  SlamMatchJacBlock other_Hf_j1 = (1 / other_rho) *
                                  other_Ca_q_G.normalized().toRotationMatrix() *
                                  other_mat;

  // Update stacked Jacobian matrices
  // current
  unsigned int row = 0;
  unsigned int col = anchor_idx * kJacCols;
  h_j.block<kSlamMatchJacRows, kJacCols>(row, kSizeCoreErr + col) =
      J_anchor_pos;

  col += n_poses_max * kJacCols;
  h_j.block<kSlamMatchJacRows, kJacCols>(row, kSizeCoreErr + col) =
      J_anchor_att;

  col = (n_poses_max * 2 + feature_id) * kJacCols;
  h_j.block<kSlamMatchJacRows, kJacCols>(row, kSizeCoreErr + col) = Hf_j1;

  // other
  col = other_anchor_idx * kJacCols;
  other_h_j.block<kSlamMatchJacRows, kJacCols>(row, kSizeCoreErr + col) =
      -other_J_anchor_pos;

  col += other_n_poses_max * kJacCols;
  other_h_j.block<kSlamMatchJacRows, kJacCols>(row, kSizeCoreErr + col) =
      -other_J_anchor_att;

  col = (other_n_poses_max * 2 + other_feature_id) * kJacCols;
  other_h_j.block<kSlamMatchJacRows, kJacCols>(row, kSizeCoreErr + col) =
      -other_Hf_j1;

  //==========================================================================
  // Outlier rejection
  //==========================================================================
  const Vectorx r_j_diag = var_landmark * Vectorx::Ones(3);
  Matrix r_j = var_landmark * Matrix::Identity(3, 3);

  Matrix S_inv = (h_j * P * h_j.transpose() +
                  other_h_j * other_P * other_h_j.transpose() + r_j)
                     .inverse();

  Matrix gamma = res_j.transpose() * S_inv * res_j;
  boost::math::chi_squared_distribution<> my_chisqr(
      3);                                 // TODO: This is a constant
  double chi = quantile(my_chisqr, 0.9);  // 95-th percentile

  if (gamma(0, 0) < chi)  // Inlier
  {
    Matrix S_j, P_j;
    // compute the innovation term and the new covariance using CI
    double w_result;
    ci.fuseCI(P, h_j, other_P, other_h_j, S_j, ci_slam_w_, w_result);
    S_j += r_j;  // add noise to S_j
    P_j = P;

    col = anchor_idx * kJacCols;
    P_j.block<kJacCols, kJacCols>(kSizeCoreErr + col, kSizeCoreErr + col) *=
        w_result;

    col += n_poses_max * kJacCols;
    P_j.block<kJacCols, kJacCols>(kSizeCoreErr + col, kSizeCoreErr + col) *=
        w_result;

    col = (n_poses_max * 2 + feature_id) * kJacCols;
    P_j.block<kJacCols, kJacCols>(kSizeCoreErr + col, kSizeCoreErr + col) *=
        w_result;

    H_list_.push_back(std::make_shared<Matrix>(h_j));
    S_list_.push_back(std::make_shared<Matrix>(S_j));  // add noise to S
    res_list_.push_back(std::make_shared<Matrix>(res_j));
    P_list_.push_back(std::make_shared<Matrix>(P_j));
  }
}