#if !defined(X_CI_H) && defined(MULTI_UAV)
#define X_CI_H

#include <cmath>
#include <iostream>
#include <memory>
#include <nlopt.hpp>
#include <vector>

#include "x/common/types.h"
#include "x/ekf/simple_state.h"
#include "x/vio/types.h"

namespace x {

class CovarianceIntersection {
 public:
  /**
   * @brief This function implements the CI algorithm according to
   * S. Julier and J. Uhlmann, “A non-divergent estimation algorithm in the
   * presence of unknown correlations,” in Proceedings of the 1997 American
   * Control Conference (Cat. No.97CH36041), vol. 4, 1997, pp. 2369–2373
   * vol.4
   *
   * @param cov_a
   * @param H_curr
   * @param state_covs
   * @param H
   * @param S
   * @param w_other
   * @param w_result
   */
  void fuseCI(const Matrix &cov_a, const Matrix &H_curr,
              const std::vector<std::shared_ptr<SimpleState>> &state_covs,
              const std::vector<Matrix> &H, Matrix &S, double w_other,
              double &w_result);

  /**
   * Same as above but for SLAM updates
   * @param cov_a
   * @param H_a
   * @param cov_b
   * @param H_b
   * @param S
   * @param w_other
   * @param w_result
   */
  void fuseCI(const Matrix &cov_a, const Matrix &H_a, const Matrix &cov_b,
              const Matrix &H_b, Matrix &S, double w_other, double &w_result);

 private:
  /**
   * Prepare the matrices for the optimization
   *
   * @param P_measure_frame
   * @param state_covs
   * @param H
   * @param w
   */
  void optimizationSetup(
      const Matrix &P_measure_frame,
      const std::vector<std::shared_ptr<SimpleState>> &state_covs,
      const std::vector<Matrix> &H, std::vector<double> &w);

  /**
   * Run the optimization problem
   *
   * @param stack_m
   * @param w
   */
  void solveW(std::vector<Matrix> &stack_m, std::vector<double> &w);

  Matrix cov_c, cov_c1, cov_c2, S_;
  Vectorx mean_c;  // this should be done also for a quaternion, right?
  Matrix stacked_cov_;
};

}  // namespace x

#endif