#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <utility>
#include <vector>

class newDSinglePairGainAnaJCostFunc : public ceres::CostFunction {
 public:
  newDSinglePairGainAnaJCostFunc(std::vector<float> o, std::vector<float> op,
                                 int num_residuals)
      : o(std::move(o)), op(std::move(op)) {
    set_num_residuals(num_residuals + 2);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
  }
  ~newDSinglePairGainAnaJCostFunc() override = default;
  bool Evaluate(double const *const *params, double *residuals,
                double **jacobians) const override {
    double aip = *params[0];
    double bip = *params[1];
    double w = 0.1;

    for (int i = 0; i < o.size(); i++) {
      residuals[i] = double(o[i]) - (double(op[i]) * (aip - bip) + bip);
    }
    residuals[o.size()] = w * (aip - 1.0);
    residuals[o.size() + 1] = w * (bip - 0.0);

    if (!jacobians) return true;

    if (jacobians[0] != nullptr)  // Jacobian for aip is requested
    {
      for (int k = 0; k < o.size(); k++) {
        jacobians[0][k] = -op[k];
      }
      jacobians[0][o.size()] = w;
      jacobians[0][o.size() + 1] = 0.0;
    }

    if (jacobians[1] != nullptr)  // Jacobian for bip is requested
    {
      for (int k = 0; k < o.size(); k++) {
        jacobians[1][k] = (op[k] - 1.0);
      }
      jacobians[1][o.size()] = 0.0;
      jacobians[1][o.size() + 1] = w;
    }
    return true;
  }

 private:
  const std::vector<float> o, op;
};

class newSinglePairGainAnaJCostFunc : public ceres::SizedCostFunction<6, 1, 1> {
 public:
  newSinglePairGainAnaJCostFunc(std::vector<float> o, std::vector<float> op)
      : o(std::move(o)), op(std::move(op)) {}
  ~newSinglePairGainAnaJCostFunc() override = default;
  bool Evaluate(double const *const *params, double *residuals,
                double **jacobians) const override {
    double aip = params[0][0];
    double bip = params[1][0];
    double w = 0.1;

    for (int i = 0; i < o.size(); i++) {
      residuals[i] = double(o[i]) - (double(op[i]) * (aip - bip) + bip);
    }
    residuals[o.size()] = w * (aip - 1.0);
    residuals[o.size() + 1] = w * (bip - 0.0);

    if (!jacobians) {
      return true;
    }

    if (jacobians[0] != nullptr)  // Jacobian for a is requested
    {
      for (int k = 0; k < o.size(); k++) {
        jacobians[0][k] = -op[k];
      }
      jacobians[0][o.size()] = w;
      jacobians[0][o.size() + 1] = 0.0;
    }

    if (jacobians[1] != nullptr)  // Jacobian for b is requested
    {
      for (int k = 0; k < o.size(); k++) {
        jacobians[1][k] = (op[k] - 1.0);
      }
      jacobians[1][o.size()] = 0.0;
      jacobians[1][o.size() + 1] = w;
    }
    return true;
  }

 private:
  const std::vector<float> o, op;
};
