#include "x/ekf/ci.h"

#include <boost/log/trivial.hpp>

using namespace x;
using namespace nlopt;

/**
 * Object function for the optimization problem
 *
 * @param w
 * @param grad
 * @param stack_m
 * @return
 */
double objFunc(const std::vector<double> &w, std::vector<double> &grad,
               void *stack_m) {
  auto *stack = (std::vector<Matrix> *)stack_m;
  Matrix result = Matrix::Zero((*stack)[0].rows(), (*stack)[0].cols());

  for (size_t i = 0; i < w.size(); i++) {
    result += w[i] * (*stack)[i];
  }
  return result.inverse().determinant();
}

/**
 * Constraint functions
 * NLopt always expects constraints to be of the form myconstraint(x) ≤ 0
 */

double constraintFuncLow(const std::vector<double> &w,
                         std::vector<double> &grad, void *data) {
  double result = 0;
  for (const double &i : w) {
    result += i;
  }
  return result - 1.0;  // == 0 is the constraint
}
double constraintFuncUp(const std::vector<double> &w, std::vector<double> &grad,
                        void *data) {
  double result = 0;
  for (const double &i : w) {
    result += i;
  }
  return 1.0 - result;  // == 0 is the constraint
}

void CovarianceIntersection::fuseCI(
    const Matrix &cov_a, const Matrix &H_curr,
    const std::vector<std::shared_ptr<SimpleState>> &state_covs,
    const std::vector<Matrix> &H, Matrix &S, const double w_other,
    double &w_result) {
  std::vector<double> w = std::vector<double>(
      H.size() + 1,
      (double)(w_other));  // they are using 0.99 for the current uav
                           // and 0.01 is distributed among the others

  if (w_other > 1.0 || w_other == 0 || w_other < -1) {
    throw std::runtime_error(
        "The CI weights must be lower than 1.0 and larger 0.0");
  }

  const auto H_size = static_cast<double>(H.size());
  if (w_other < 0.0) {
    w = std::vector<double>(H.size() + 1, w_other);
    w[0] = 1.0 - H_size * w_other;
    Matrix P_measure_frame = H_curr * cov_a.inverse() * H_curr.transpose();
    optimizationSetup(P_measure_frame, state_covs, H, w);
    if (w.empty()) {  // in case the nlopt fails we use the fixed w
      w = std::vector<double>(H.size() + 1, (double)(w_other));
      w[0] = 1.0 - H_size * w_other;
    }
  } else {
    w[0] = 1.0 - H_size * w_other;
  }

  S_ = (1.0 / w[0]) * H_curr * cov_a * H_curr.transpose();

  for (size_t i = 0; i < state_covs.size(); i++) {
    S_ += (1.0 / w[i + 1]) * H[i] * (state_covs[i]->getCovariance()) *
          H[i].transpose();  // + noise....
  }
  // update S only if needed
  S = S_;

  BOOST_LOG_TRIVIAL(debug) << "WEIGHT P_ =" << w[1] << std::endl;
  BOOST_LOG_TRIVIAL(debug) << "S : " << S << std::endl;

  w_result = 1.0 / w[0];
  BOOST_LOG_TRIVIAL(debug) << "--------> multi MSCKF CI performed" << std::endl;
}

void CovarianceIntersection::fuseCI(const Matrix &cov_a, const Matrix &H_a,
                                    const Matrix &cov_b, const Matrix &H_b,
                                    Matrix &S, double w_other,
                                    double &w_result) {
  if (w_other > 1.0 || w_other == 0 || w_other < -1) {
    throw std::runtime_error(
        "The CI weights must be lower than 1.0 and larger than 0.0");
  }
  Matrix P_a = H_a * cov_a * H_a.transpose();  // current UAV
  Matrix P_b = H_b * cov_b * H_b.transpose();  // other UAV

  if (w_other < 0.0) {
    std::vector<Matrix> stack_m;
    stack_m.emplace_back(H_a * cov_a.inverse() * H_a.transpose());
    stack_m.emplace_back(H_b * cov_b.inverse() * H_b.transpose());
    std::vector<double> w =
        std::vector<double>(2);  // in case the nlopt fails we use the fixed w
    w[0] = 1.0 + w_other;
    w[1] = -w_other;
    solveW(stack_m, w);
    if (w.empty()) {
      w_other = -w_other;
    } else {
      w_other = w[1];
    }
  }
  w_result = (1.0 / (1.0 - w_other));

  S_ = (1.0 / (1.0 - w_other)) * P_a + (1.0 / w_other) * P_b;

  S = S_;

  BOOST_LOG_TRIVIAL(debug) << "--------> multi SLAM CI performed " << std::endl;
}

void CovarianceIntersection::optimizationSetup(
    const Matrix &P_measure_frame,
    const std::vector<std::shared_ptr<SimpleState>> &state_covs,
    const std::vector<Matrix> &H, std::vector<double> &w) {
  std::vector<Matrix> stack_m;
  stack_m.emplace_back(P_measure_frame);

  for (size_t i = 0; i < state_covs.size(); i++) {
    stack_m.emplace_back(H[i] * state_covs[i]->getCovariance().inverse() *
                         H[i].transpose());
  }
  solveW(stack_m, w);
}

void CovarianceIntersection::solveW(std::vector<Matrix> &stack_m,
                                    std::vector<double> &w) {
  opt opt(LN_COBYLA, stack_m.size());

  double lb = 1e-4, ub = 1.0;
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_maxtime(0.1);
  opt.set_min_objective(objFunc, &stack_m);

  opt.add_inequality_constraint(constraintFuncUp, &lb, 1e-6);
  opt.add_inequality_constraint(constraintFuncLow, &lb, 1e-6);

  opt.set_ftol_abs(1e-6);

  double minf;

  try {
    nlopt::result result = opt.optimize(w, minf);
    if (result >= nlopt::result::SUCCESS) {
      double sum = 0.0;
      for (double const e : w) {
        sum += e;
      }
      for (double &e : w) {
        e /= sum;
        BOOST_LOG_TRIVIAL(info) << "e : " << e << std::endl;
      }

      // #ifdef DEBUG
      BOOST_LOG_TRIVIAL(info)
          << "--------> Optimization result : " << result << std::endl
          << "--------> found minimum at f(" << w[0] << "," << w[1]
          << ") = " << minf << std::endl;
      // #endif
    } else {
      BOOST_LOG_TRIVIAL(info)
          << "--------> Optimization result : " << result << std::endl;
      w.clear();
    }

  } catch (std::exception &e) {
    // #ifdef DEBUG
    BOOST_LOG_TRIVIAL(error) << "NLopt failed: " << e.what() << std::endl;
    w.clear();
    // #endif
  }
}
