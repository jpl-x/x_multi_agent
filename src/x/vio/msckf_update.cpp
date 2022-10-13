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

#include "x/vio/msckf_update.h"

#include <boost/log/trivial.hpp>
#include <boost/math/distributions.hpp>
#include <memory>

#include "x/vio/tools.h"

using namespace x;

MsckfUpdate::MsckfUpdate(const x::TrackList &trks, const x::AttitudeList &quats,
                         const x::TranslationList &poss, const Matrix &cov_s,
                         const int n_poses_max, const double sigma_img
#ifdef MULTI_UAV
                         ,
                         MsckfMatches &tracks_matches, const double ci_msckf_w
#endif
) {
#ifdef MULTI_UAV
  ci_msckf_w_ = ci_msckf_w;
#endif
  // Number of features
  const int n_trks = static_cast<int>(trks.size());

  // Number of feature observations
  int n_obs = 0;
  for (int i = 0; i < n_trks; i++) {
    n_obs += static_cast<int>(trks[i].size());
  }

  // Initialize Kalman update matrices
  const int rows = 2 * n_obs - n_trks * 3;
  const int cols = static_cast<int>(cov_s.cols());
  jac_ = Matrix::Zero(rows, cols);
  cov_m_diag_ = Vectorx::Ones(rows);
  res_ = Matrix::Zero(rows, 1);

  // For each track, compute residual, Jacobian and covariance block
  const double var_img = sigma_img * sigma_img;
  for (int i = 0, row_h = 0; i < n_trks; ++i) {
    preProcessOneTrack(trks[i], quats, poss,
#ifdef MULTI_UAV
                       tracks_matches,
#endif
                       cov_s, n_poses_max, var_img, i, row_h);
  }
}

void MsckfUpdate::preProcessOneTrack(const x::Track &track,
                                     const x::AttitudeList &C_q_G,
                                     const x::TranslationList &G_p_C,
#ifdef MULTI_UAV
                                     MsckfMatches &tracks_matches,
#endif
                                     const Matrix &P, const int n_poses_max,
                                     const double var_img, const int &j,
                                     int &row_h) {
  /** copy the track, the translations and attitudes, we will modify them to
   * perform a multi MSCKFupdate
   */
  Track tmp_track = Track();
  TranslationList tmp_G_p_C = TranslationList();
  AttitudeList tmp_C_q_G = AttitudeList();

  /** clear all the references **/
  std::vector<std::shared_ptr<MsckfMatch>> tracks_matches_reference_;
  std::vector<std::shared_ptr<SimpleState>> state_cov_reference_;
  std::vector<std::shared_ptr<TranslationList>> translation_list_reference_;
  std::vector<std::shared_ptr<AttitudeList>> attitude_list_reference_;
  std::vector<int> state_size_list_;

  int multi_cols = static_cast<int>(P.cols());

#ifdef MULTI_UAV
  state_size_list_.emplace_back(P.cols());

  // iterate over the tracks received and find correspondences
  int corrected_index = 0;

  for (int i = 0; i < static_cast<int>(tracks_matches.size()); i++) {
    if (track.getId() == tracks_matches[i - corrected_index].id_current_track) {
      tracks_matches_reference_.emplace_back(std::make_shared<MsckfMatch>(
          tracks_matches[i - corrected_index]));  // keep track of the matches

      // keep track of the matches' state and covariance
      std::shared_ptr<x::SimpleState> sptr_sc =
          tracks_matches[i - corrected_index].state;
      state_cov_reference_.push_back(sptr_sc);

      // keep track of the matches' state length, needed to divide the
      // jacobians in the stacked matrix
      state_size_list_.push_back(
          state_cov_reference_.back()->getErrorStateSize());
      multi_cols += state_cov_reference_.back()->getErrorStateSize();

      // Insert match' positions will use these to triangulate the feature
      TranslationList ts_l =
          state_cov_reference_.back()->getCameraPositionsList();
      tmp_G_p_C.insert(
          tmp_G_p_C.end(),
          ts_l.end() - static_cast<int>(tracks_matches[i - corrected_index]
                                            .received_track_ptr->size()),
          ts_l.end());

      // Insert match' attitudes will use these to triangulate the feature
      AttitudeList at_l = state_cov_reference_.back()->getCameraAttitudesList();
      tmp_C_q_G.insert(
          tmp_C_q_G.end(),
          at_l.end() - static_cast<int>(tracks_matches[i - corrected_index]
                                            .received_track_ptr->size()),
          at_l.end());

      // concatenate the tracks, needed for triangulation
      tmp_track.insert(
          tmp_track.end(),
          tracks_matches[i - corrected_index].received_track_ptr->begin(),
          tracks_matches[i - corrected_index].received_track_ptr->end());

      // remove the match from the list
      tracks_matches.erase(tracks_matches.begin() + i - corrected_index);
      corrected_index++;
    }
  }
  // check wheter something is inconsistent
  assert(tracks_matches_reference_.size() == state_cov_reference_.size());

#endif

  tmp_G_p_C.insert(tmp_G_p_C.end(),
                   G_p_C.end() - static_cast<int>(track.size()), G_p_C.end());
  tmp_C_q_G.insert(tmp_C_q_G.end(),
                   C_q_G.end() - static_cast<int>(track.size()), C_q_G.end());
  tmp_track.insert(tmp_track.end(), track.begin(), track.end());

  // Initialize the Jacobians and residual matrices for the multi UAV update
  const int multi_rows =
      3 * (static_cast<int>(tracks_matches_reference_.size()) + 1);
  jac_x_pf_ = Matrix::Zero(multi_rows, multi_cols);
  jac_pf_ = Matrix::Zero(multi_rows, 3);
  res_pf_ = Matrix::Zero(multi_rows, 1);

  // Triangulate the landmark with all the available poses and observations
  Vector3 feature;
  Triangulation triangulator(tmp_C_q_G, tmp_G_p_C, max_iter_, term_);
  triangulator.triangulateGN(tmp_track, feature);

  Vector3 global_feature_pose;
  getGlobalFeaturePosition(feature, tmp_C_q_G.back(), tmp_G_p_C.back(),
                           global_feature_pose);

  // Compute the jacobians of the current UAV
  int multi_row_h = 0, multi_col_h = 0;
#ifdef MULTI_UAV
  proceed_with_multi_ = false;
#endif
  processOneTrack(track, P, C_q_G, G_p_C, n_poses_max, var_img,
                  global_feature_pose, multi_row_h, multi_col_h, row_h, false);

#ifdef MULTI_UAV
  // If we have correspondences between tracks then we can perform a
  // Multi UAV update
  bool is_multi_uav = !tracks_matches_reference_.empty();

  if (proceed_with_multi_ && is_multi_uav) {
    BOOST_LOG_TRIVIAL(debug)
        << "State Cov size : " << state_cov_reference_.size() << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "tracks_matches_reference size : "
                             << tracks_matches_reference_.size() << std::endl;

    //  We need to consider the measurements that depend on the feature point Pf
    // iterate over the matches and compute the jacobinas and the residuals
    for (size_t i = 0; i < tracks_matches_reference_.size(); i++) {
      int n_poses = state_cov_reference_[i]->nPosesMax();

      // Compute the Jacobians and the residuals for each received track. The
      // Jacobian that depends only on the feature point Pf
      int not_used_var = 0;
      multi_row_h++;
      const Track trk = *(tracks_matches_reference_[i]->received_track_ptr);
      const Matrix cov = state_cov_reference_[i]->getCovariance();
      const AttitudeList attitudes =
          state_cov_reference_[i]->getCameraAttitudesList();
      const TranslationList translations =
          state_cov_reference_[i]->getCameraPositionsList();
      processOneTrack(trk, cov, attitudes, translations, n_poses, var_img,
                      global_feature_pose, multi_row_h, multi_col_h,
                      not_used_var, true);
    }

    // Compute the null space projection for the feature point
    nullSpaceProjection(jac_x_pf_, jac_pf_,
                        res_pf_);  // from here we can use jac_x_pf_ and
                                   // res_msckf to perform a multi msckf update

    // Split the full Jacobian in partial Jacobians referring to their state
    std::vector<Matrix> Hs;
    Matrix h_j = jac_x_pf_.block(0, 0, jac_x_pf_.rows(), state_size_list_[0]);

    Matrix S_j,  // Needed by the EKF update
        P_j;     // needed by the EKF update for multi UAV
    S_j = h_j * P * h_j.transpose();

    for (int i = 1, h_col = state_size_list_[0];
         i < static_cast<int>(state_size_list_.size()); i++) {
      Hs.emplace_back(
          jac_x_pf_.block(0, h_col, jac_x_pf_.rows(), state_size_list_[i]));
      h_col += state_size_list_[i];

      S_j += Hs[i - 1] * state_cov_reference_[i - 1]->getCovariance() *
             Hs[i - 1].transpose();
    }

    // Compute the cov intersection:
    // S^(-1) = Sum(w_i*(H_i*P_i*H_i^T)^(-1)) for all i

    BOOST_LOG_TRIVIAL(debug)
        << "State size list : " << state_size_list_.size() << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "Hs size : " << Hs.size() << std::endl;

    // Add noise to the S matrix
    S_j += var_img * Matrix::Identity(S_j.cols(), S_j.cols());  // noise;
    assert(S_j.cols() == 3 * tracks_matches_reference_.size());

    //==========================================================================
    // Outlier rejection
    //==========================================================================
    Matrix S_inv = S_j.inverse();
    Matrix gamma = res_pf_.transpose() * S_inv * res_pf_;
    const boost::math::chi_squared_distribution<> my_chisqr(
        2.0 * static_cast<double>(tmp_track.size()) - 3.0);  // 2*Mj-3 DOFs
    const double chi = quantile(my_chisqr, 0.95);            // 95-th percentile

    // Store all the computed values. They will be used in the EKF update.
    if (gamma(0, 0) < chi) {
      double w_result;
      cov_intersection_.fuseCI(P, h_j, state_cov_reference_, Hs, S_j,
                               ci_msckf_w_, w_result);

      S_j += var_img * Matrix::Identity(S_j.cols(), S_j.cols());
      P_j = P;

      for (size_t i = 0; i < track.size(); ++i) {
        unsigned int pos = C_q_G.size() - track.size() + i;
        unsigned int col = pos * kJacCols;
        P_j.block<kJacCols, kJacCols>(kSizeCoreErr + col, kSizeCoreErr + col) *=
            w_result;

        col += n_poses_max * kJacCols;
        P_j.block<kJacCols, kJacCols>(kSizeCoreErr + col, kSizeCoreErr + col) *=
            w_result;
      }

      S_list_.emplace_back(std::make_shared<Matrix>(S_j));
      P_list_.emplace_back(std::make_shared<Matrix>(P_j));
      H_list_.emplace_back(std::make_shared<Matrix>(h_j));
      res_list_.emplace_back(std::make_shared<Matrix>(res_pf_));
      BOOST_LOG_TRIVIAL(debug)
          << ">>>>>>>>>>>>>>>>>>>>>>>>READY FOR MULTI MSCKF "
             "UPDATE<<<<<<<<<<<<<<<<<<<<<<<<<"
          << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "MULTI MSCKF" << std::endl;
    }
  }
#endif
}

void MsckfUpdate::getGlobalFeaturePosition(const Vector3 &inverse_depth,
                                           const Attitude &C_q_G,
                                           const Translation &G_p_C,
                                           Vector3 &global_feature_pose) {
  // Inverse-depth parameters in last observation frame
  const double alpha = inverse_depth(0);
  const double beta = inverse_depth(1);
  const double rho = inverse_depth(2);

  // Coordinate of feature in global frame
  x::Quaternion Cn_q_G;
  Cn_q_G.x() = C_q_G.ax;
  Cn_q_G.y() = C_q_G.ay;
  Cn_q_G.z() = C_q_G.az;
  Cn_q_G.w() = C_q_G.aw;

  Vector3 G_p_Cn(G_p_C.tx, G_p_C.ty, G_p_C.tz);

  global_feature_pose = 1.0 / (rho)*Cn_q_G.normalized().toRotationMatrix() *
                            Vector3(alpha, beta, 1) +
                        G_p_Cn;  // estimated feature position in world frame
}

void MsckfUpdate::processOneTrack(const x::Track &track, const Matrix &P,
                                  const x::AttitudeList &C_q_G,
                                  const x::TranslationList &G_p_C,
                                  const int n_poses_max, const double var_img,
                                  const Vector3 &G_p_fj, int &multi_row_h,
                                  int &multi_col_h, int &row_h,
                                  bool is_multi_msckf) {
  // Initialization
  const int track_size = static_cast<int>(track.size());

  // n of observation in the track * 2 (x,y)
  const int rows_track_j = track_size * 2;

  // state length 15+6*m+3*N
  const int cols = static_cast<int>(P.cols());
  Matrix jac_j(Matrix::Zero(rows_track_j, cols));
  Matrix Hf_j(Matrix::Zero(rows_track_j, kJacCols));
  Matrix res_j(Matrix::Zero(rows_track_j, 1));

  x::Quatern attitude_to_quaternion;

  // LOOP OVER ALL FEATURE OBSERVATIONS
  for (int i = 0; i < track_size; ++i) {
    const unsigned int pos =
        G_p_C.size() - track_size + i;  // it could be that the track length is
                                        // lower than the sliding window.

    // position of the camera that observed this feature
    Quaternion Ci_q_G_ = attitude_to_quaternion(C_q_G[pos]);
    Vector3 G_p_Ci_(G_p_C[pos].tx, G_p_C[pos].ty, G_p_C[pos].tz);

    // Feature position expressed in camera frame.
    Vector3 Ci_p_fj;
    Ci_p_fj << Ci_q_G_.normalized().toRotationMatrix().transpose() *
                   (G_p_fj - G_p_Ci_);

    // eq. 20(a)
    Eigen::Vector2d z;
    z(0) = track[i].getX();
    z(1) = track[i].getY();

    Eigen::Vector2d z_hat(z);
    assert(Ci_p_fj(2));
    if (!((Ci_p_fj.array() == Ci_p_fj.array())).all()) {
#ifdef MULTI_UAV
      proceed_with_multi_ = false;
#endif
#ifdef VERBOSE
      outliers_.push_back(G_p_fj);
#endif
      return;
      // throw std::runtime_error("Ci_p_fj is 0");
    }
    z_hat(0) = Ci_p_fj(0) / Ci_p_fj(2);
    z_hat(1) = Ci_p_fj(1) / Ci_p_fj(2);

    // eq. 20(b)
    res_j(i * 2, 0) = z(0) - z_hat(0);
    res_j(i * 2 + 1, 0) = z(1) - z_hat(1);

    // Set Jacobian of pose for i'th measurement of feature j (eq.22, 23)
    VisJacBlock J_i = VisJacBlock::Zero();
    // first row
    J_i(0, 0) = 1.0 / Ci_p_fj(2);  // 1/z
    J_i(0, 1) = 0.0;               // it's already 0.0
    J_i(0, 2) = -Ci_p_fj(0) / std::pow((double)Ci_p_fj(2), 2);  // x/z^2
    // second row
    J_i(1, 0) = 0.0;  // it's already 0.0
    J_i(1, 1) = 1.0 / Ci_p_fj(2);
    J_i(1, 2) = -Ci_p_fj(1) / std::pow((double)Ci_p_fj(2), 2);  // y/z^2

    unsigned int row = i * kVisJacRows;

    // Measurement Jacobians wrt attitude, position and feature
    // Position
    VisJacBlock J_position =
        -J_i * Ci_q_G_.normalized().toRotationMatrix().transpose();

    Vector3 dP = Ci_q_G_.normalized().toRotationMatrix().transpose() *
                 (G_p_fj - G_p_Ci_);

    // Attitude
    VisJacBlock J_attitude = J_i * x::Skew(dP(0), dP(1), dP(2)).matrix;

    // Feature
    // Hf_j.block<kVisJacRows, kJacCols>(row, 0) = -J_position;

    const Vector3 g_ = Vector3(0.0, 0.0, -9.81);
    // avoid the observability issue on the yaw.
    // Observability-constrained Vision-aided Inertial Navigation, Hesch J. et
    // al. Feb, 2012
    Matrix A_pos = J_position;
    Matrix u_pos = Ci_q_G_.normalized().toRotationMatrix() * g_;
    J_position = A_pos - A_pos * u_pos * (u_pos.transpose() * u_pos).inverse() *
                             u_pos.transpose();

    Matrix A_att = J_attitude;
    Vector3 vec = G_p_fj - G_p_Ci_;
    Matrix u_att = x::Skew(vec(0), vec(1), vec(2)).matrix * g_;
    J_attitude = A_att - A_att * u_att * (u_att.transpose() * u_att).inverse() *
                             u_att.transpose();

    // Feature
    Hf_j.block<kVisJacRows, kJacCols>(row, 0) = -J_position;

    // Update stacked Jacobian matrix associated to the current feature
    unsigned int col = pos * kJacCols;
    jac_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_position;

    col += n_poses_max * kJacCols;
    jac_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_attitude;
  }  // LOOP OVER ALL FEATURE OBSERVATIONS

  //========================================================================
  // Left nullspace projection
  //========================================================================
  // Nullspace computation
  Matrix q = Hf_j.householderQr().householderQ();
  Matrix A_up =
      q.block(0, 0, q.rows(), 3);  // needed for the part of the system
                                   // that depends on the feature point
  Matrix A = x::MatrixBlock(q, 0, 3);

  // Projections of the part of the system that does not depend on the feature
  // point
  Matrix res0_j = A.transpose() * res_j;
  Matrix jac0_j = A.transpose() * jac_j;

#ifdef MULTI_UAV
  // if we are performing the multi MSCKF
  // we check if it is an inlier or an outlier on the stacked jacobians
  // but we don't save the jacobians regarding the state only.
  // Here we save the part of the jacobian that depends on the feature point
  jac_x_pf_.block(3 * multi_row_h, multi_col_h, 3, cols) =
      A_up.transpose() * jac_j;
  jac_pf_.block<kJacCols, kJacCols>(3 * multi_row_h, 0) =
      A_up.transpose() * Hf_j;
  res_pf_.block<kJacCols, 1>(3 * multi_row_h, 0) = A_up.transpose() * res_j;
  multi_col_h += cols;

  if (is_multi_msckf) {
    return;
  }
#endif

  // Noise measurement matrix
  const Vectorx r0_j_diag = var_img * Vectorx::Ones(rows_track_j - 3);
  const Matrix r0_j = r0_j_diag.asDiagonal();
  //==========================================================================
  // Outlier rejection
  //==========================================================================
  Matrix S_inv = (jac0_j * P * jac0_j.transpose() + r0_j).inverse();
  Matrix gamma = res0_j.transpose() * S_inv * res0_j;
  boost::math::chi_squared_distribution<> my_chisqr(2.0 * track_size -
                                                    3.0);  // 2*Mj-3 DOFs
  double chi = quantile(my_chisqr, 0.95);                  // 95-th percentile

  if (gamma(0, 0) < chi) {
    // Inlier
#ifdef VERBOSE
    inliers_.push_back(G_p_fj);
#endif
    jac_.block(row_h,             // startRow
               0,                 // startCol
               rows_track_j - 3,  // numRows
               cols) = jac0_j;    // numCols

    // Residual vector (for this track)
    res_.block(row_h, 0, rows_track_j - 3, 1) = res0_j;

    // Measurement covariance matrix diagonal
    cov_m_diag_.segment(row_h, rows_track_j - 3) = r0_j_diag;

    row_h += rows_track_j - 3;
#ifdef MULTI_UAV
    proceed_with_multi_ = true;
#endif
  } else {
    // outlier
#ifdef MULTI_UAV
    proceed_with_multi_ = false;
#endif
#ifdef VERBOSE
    outliers_.push_back(G_p_fj);
#endif
  }
}

void MsckfUpdate::nullSpaceProjection(Matrix &H_j, Matrix &Hf_j, Matrix &res) {
  // Nullspace computation
  const Matrix q = Hf_j.householderQr().householderQ();
  const Matrix A = x::MatrixBlock(q, 0, 3);

  H_j = A.transpose() * H_j;
  res = A.transpose() * res;
}