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

#include "x/ekf/updater.h"

using namespace x;

#ifdef MULTI_UAV
void Updater::collaborativeUpdate(State &state) {
  // Image pre-processing
  // check if there are SLAM matches.
  if (preUpdateCI()) {
    std::vector<std::shared_ptr<Matrix>> S_list, P_list, H_list, res_list;

    // Construct Jacobian, residual and noise covariance matrices
    constructSlamCIUpdate(state, S_list, P_list, H_list, res_list);
    // Apply CI update
    for (size_t i = 0; i < P_list.size(); i++) {
      applyCI(state, *(P_list[i].get()), *(H_list[i].get()),
              *(res_list[i].get()), *(S_list[i].get()));
    }
  }
}
#endif

void Updater::update(State &state) {
  // Image pre-processing

  Matrix h, res, r;
  Matrix correction = Matrix::Zero(state.nErrorStates(), 1);

  // track features, add last estimated pose to the state_manager for MSCKF
  // baseline estimation.
  // NB. this is not adding anything to the state poses
  preProcess(state);

  // before sliding the pose with the new prior, we need to update the short
  // MSCKF tracks
  const bool short_update_requested = preUpdateShortMsckf();
  if (short_update_requested) {
    // true if this is the last loop iteration
#ifdef MULTI_UAV
    std::vector<std::shared_ptr<Matrix>> S_list, P_list, H_list, res_list;

    // Construct Jacobian, residual and noise covariance matrices
    constructShortMsckfUpdate(state, h, res, r, S_list, P_list, H_list,
                              res_list);

    // Apply CI update
    for (size_t j = 0; j < P_list.size(); j++) {
      applyCI(state, *(P_list[j].get()), *(H_list[j].get()),
              *(res_list[j].get()), *(S_list[j].get()));
    }
#else
    // Construct Jacobian, residual and noise covariance matrices
    constructShortMsckfUpdate(state, h, res, r);
    if (h.size() > 0) {
      // Apply update
      applyUpdate(state, h, res, r, correction, true);
    }
#endif
  }

  // Pre-update work and check if update is needed
  const bool update_requested = preUpdate(state);

  if (update_requested) {
    // Initialize correction vector
    correction = Matrix::Zero(state.nErrorStates(), 1);

#ifdef MULTI_UAV
    std::vector<std::shared_ptr<Matrix>> S_list, P_list, H_list, res_list;
    // Construct Jacobian, residual and noise covariance matrices
    constructUpdate(state, h, res, r, S_list, P_list, H_list, res_list);

    // Apply CI update
    for (size_t j = 0; j < P_list.size(); j++) {
      applyCI(state, *(P_list[j].get()), *(H_list[j].get()),
              *(res_list[j].get()), *(S_list[j].get()));
    }
    if (h.size() > 0) {
      // Apply CI update
      applyUpdate(state, h, res, r, correction, true);
    }
#else
    for (int i = 0; i < iekf_iter_; i++) {
      const bool is_last_iter =
          i == iekf_iter_ - 1;  // true if this is the last loop iteration

      // Construct Jacobian, residual and noise covariance matrices
      constructUpdate(state, h, res, r);

      if (h.size() > 0) {
        // Apply update
        applyUpdate(state, h, res, r, correction, is_last_iter);
      }
    }
#endif
    // Post update: feature initialization
    postUpdate(state, correction);
  }
}

void Updater::applyUpdate(State &state, const Matrix &H, const Matrix &res,
                          const Matrix &R, Matrix &correction_total,
                          const bool cov_update) {
  // Compute Kalman gain and state correction
  // TODO(jeff) Assert state correction doesn't have NaNs/Infs.
  Matrix &P = state.getCovarianceRef();

  Matrix S = H * P * H.transpose() + R;
  const Matrix K = P * H.transpose() * S.inverse();
  Matrix correction = K * (res + H * correction_total) - correction_total;

  // Covariance update (skipped if this is not the last IEKF iteration)
  const auto n = P.rows();
  if (cov_update) {
    P = (Matrix::Identity(n, n) - K * H) * P;
    // Make sure P stays symmetric.
    P = 0.5 * (P + P.transpose());
  }

  // State update
  state.correct(correction);

  // Add correction at current iteration to total (for IEKF)
  correction_total += correction;
}

#ifdef MULTI_UAV
void Updater::applyCI(State &state, Matrix &ci_P, const Matrix &H,
                      const Matrix &res, Matrix &S) {
  // Compute Kalman gain and state correction with the CI covariance
  // TODO(jeff) Assert state correction doesn't have NaNs/Infs.
  Matrix &P = state.getCovarianceRef();

  const Matrix K = ci_P * H.transpose() * S.inverse();

  Matrix correction = K * res;

  const auto n = P.rows();
  P = (Matrix::Identity(n, n) - K * H) * ci_P;
  // Make sure P stays symmetric.
  P = 0.5 * (P + P.transpose());

  // State update
  state.correct(correction);
}
#endif