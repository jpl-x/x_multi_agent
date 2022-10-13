/*
 * Author: Manash Pratim Das (mpdmanash@iitkgp.ac.in)
 *
 */

#include "x/photometric_calibration/irPhotoCalib.h"

#include <ceres/ceres.h>

#include <boost/log/trivial.hpp>
#include <utility>

#include "x/photometric_calibration/photoetricOptimization.h"

IRPhotoCalib::IRPhotoCalib(int w, int h, int k_div, bool k_calibrate_SP,
                           float k_SP_threshold, double epsilon_gap,
                           double epsilon_base, bool use_keyframes)
    : use_keyframes_{use_keyframes},
      div_{k_div},
      w_{w},
      h_{h},
      epsilon_gap_{epsilon_gap},
      epsilon_base_{epsilon_base} {
  omp_set_num_threads(4);
  params_PT_.push_back({1.0, 0.0});

  sids_history_ = std::vector<int>();
  sids_current_ = std::vector<int>();

  // Spatial Params
  spatial_coverage_ =
      cv::Mat(cv::Size(static_cast<int>((h_ / div_) * (w_ / div_)), 1), CV_8UC1,
              cv::Scalar(0));
  spatial_coverage_.at<uchar>(
      0, getNid(static_cast<int>(w_ / 2), static_cast<int>(h_ / 2)));
  params_PS_ = cv::Mat(cv::Size(w_, h_), CV_32FC1, cv::Scalar(0.f));
  calibrate_SP_ = k_calibrate_SP;
  SP_threshold_ = k_SP_threshold;
  SP_correscount_ = std::vector<int>((w_ * h_) / (div_ * div_), 0);
  SP_max_correscount_ = 500;

  lut_ = cv::Mat(cv::Size(256, 1), CV_8UC1, cv::Scalar(0));
  for (int i = 0; i < 256; i++) {
    if (i < 128) {
      lut_.at<uchar>(0, i) = (uchar)i * 2;
    } else if (i == 128) {
      lut_.at<uchar>(0, i) = (uchar)255;
    } else {
      lut_.at<uchar>(0, i) = (uchar)(512 - 2 * i);
    }
  }

  GP_length_scale_ = 5;
  GP_sigma_f_ = 0.01;
  GP_sigma_n_ = 0.01;
}

IRPhotoCalib::~IRPhotoCalib() { spatial_par_thread_.join(); }

PTAB IRPhotoCalib::getPrevAB() {
  if (use_keyframes_) {
    return params_PT_[latest_KF_id_];
  } else [[likely]] {
    return params_PT_[frame_id_];
  }
}

void IRPhotoCalib::getRelativeGains(const double a1, const double b1,
                                    const double a2, const double b2,
                                    double &a12, double &b12) {
  double e12 = (a2 - b2) / (a1 - b1);
  b12 = (b2 - b1) / (a1 - b1);
  a12 = e12 + b12;
}

void IRPhotoCalib::chainGains(const double a01, const double b01,
                              const double a12, const double b12, double &a02,
                              double &b02) {
  double e02 = (a01 - b01) * (a12 - b12);
  b02 = b01 + (a01 - b01) * b12;
  a02 = e02 + b02;
}

int IRPhotoCalib::getNid(const int ptx, const int pty) const {
  return static_cast<int>(std::floor(pty / div_) * std::floor(w_ / div_) +
                          std::floor(ptx / div_));
}

std::pair<int, int> IRPhotoCalib::getInvNid(const int sid) const {
  int pty = static_cast<int>(sid / static_cast<int>(std::floor(w_ / div_)));
  int ptx = static_cast<int>(sid % static_cast<int>(std::floor(w_ / div_)));
  return std::make_pair(ptx, pty);
}

void IRPhotoCalib::ProcessCurrentFrame(
    std::vector<std::vector<float>> &intensity_history,
    std::vector<std::vector<float>> &intensity_current,
    const std::vector<int> &frame_ids_history,
    std::vector<std::vector<std::pair<int, int>>> &pixels_history,
    std::vector<std::vector<std::pair<int, int>>> &pixels_current,
    bool thisKF) {
  std::unique_lock<std::mutex> lck(photo_mtx_);
  // Match this frame with previous frames
  PTAB prevAB = getPrevAB();
  double a_origin_previous = prevAB.a;
  double b_origin_previous = prevAB.b;
  double w_a = 0;
  double w_b = 0;
  int w_count = 0;

  auto iter_params_PT = params_PT_.begin();
#pragma omp parallel for default(none) shared(               \
    intensity_history, intensity_current, frame_ids_history, \
    a_origin_previous, b_origin_previous, w_a, w_b, w_count, iter_params_PT)
  for (int i = 0; i < intensity_history.size(); i++) {
    if (intensity_history[i].size() <= 4) {
      continue;
    }

    iter_params_PT = params_PT_.begin() + frame_id_ + 1 - frame_ids_history[i];

    double a_history_current, b_history_current, a_origin_current,
        b_origin_current, a_previous_current, b_previous_current;
    int support_points =
        EstimateGainsRansac(intensity_history[i], intensity_current[i],
                            a_history_current, b_history_current);
    double a_origin_history = iter_params_PT->a;
    // this->params_PT_[this->frame_id_ + 1 - frame_ids_history[i]].a;
    double b_origin_history = iter_params_PT->b;
    // this->params_PT_[this->frame_id_ + 1 - frame_ids_history[i]].b;
    chainGains(a_origin_history, b_origin_history, a_history_current,
               b_history_current, a_origin_current, b_origin_current);
    getRelativeGains(a_origin_previous, b_origin_previous, a_origin_current,
                     b_origin_current, a_previous_current,
                     b_previous_current);  // May be only do it previous key
                                           // frame and not previour frame
    w_a += a_previous_current * support_points;
    w_b += b_previous_current * support_points;
    w_count += support_points;
  }
  double w_a_previous_current = w_a / w_count;
  double w_b_previous_current = w_b / w_count;
  if (w_count < 5) {
    w_a_previous_current = 1.0;
    w_b_previous_current = 0.0;
  }  // in case, we do not have enough correspondence to estimate AB

  // Drift adjustment
  double delta =
      (1.0 - (w_a_previous_current - w_b_previous_current)) * epsilon_gap_;
  w_a_previous_current = w_a_previous_current + delta;
  w_b_previous_current = w_b_previous_current - delta;
  w_a_previous_current =
      w_a_previous_current - (w_a_previous_current - 1.0) * epsilon_base_;
  w_b_previous_current =
      w_b_previous_current - (w_b_previous_current)*epsilon_base_;

  double a_origin_current, b_origin_current;
  chainGains(a_origin_previous, b_origin_previous, w_a_previous_current,
             w_b_previous_current, a_origin_current, b_origin_current);

  // Spatial Calibration
  if (calibrate_SP_) {
    for (size_t i = 0; i < pixels_current.size(); i++) {
      iter_params_PT =
          params_PT_.begin() + frame_id_ + 1 - frame_ids_history[i];
      const double a_origin_history = iter_params_PT->a;
      const double b_origin_history = iter_params_PT->b;
      double a_history_current, b_history_current;
      getRelativeGains(a_origin_history, b_origin_history, a_origin_current,
                       b_origin_current, a_history_current, b_history_current);

      int sid_history = 0;
      int sid_current = 0;
      for (size_t j = 0; j < pixels_current[i].size(); j++) {
        sid_history =
            getNid(pixels_history[i][j].first, pixels_history[i][j].second);
        sid_current =
            getNid(pixels_current[i][j].first, pixels_current[i][j].second);
        if (sid_history == sid_current ||
            (SP_correscount_[sid_history] > SP_max_correscount_ &&
             SP_correscount_[sid_history] > SP_max_correscount_)) {
          continue;
        }

        spatial_coverage_.at<uchar>(0, sid_history) = 1;
        spatial_coverage_.at<uchar>(0, sid_current) = 1;

        sids_history_.push_back(sid_history);
        sids_current_.push_back(sid_current);

        m_SP_vecB.emplace_back(intensity_current[i][j] *
                                   (a_history_current - b_history_current) -
                               intensity_history[i][j] + b_history_current);
        SP_correscount_[sid_current]++;
        SP_correscount_[sid_history]++;
      }
    }
    static const auto tot = spatial_coverage_.rows * spatial_coverage_.cols;
    double coverage_ratio =
        cv::norm(spatial_coverage_, cv::NORM_HAMMING) / (tot);
    if (coverage_ratio > SP_threshold_) {
      calibrate_SP_ = false;
      spatial_par_thread_ =
          std::thread(&IRPhotoCalib::EstimateSpatialParameters, this, w_, h_,
                      div_, GP_length_scale_, GP_sigma_f_, GP_sigma_n_,
                      m_SP_vecB, sids_history_, sids_current_);
      // spatial_par_thread_.detach();
    }
  }

  frame_id_++;
  PTAB this_frame_params{a_origin_current, b_origin_current};
  params_PT_.push_back(this_frame_params);
  if (params_PT_.size() > 15) {
    params_PT_.erase(params_PT_.begin());
    frame_id_--;
  }
}

int IRPhotoCalib::EstimateGainsRansac(std::vector<float> oi,
                                      std::vector<float> opip, double &out_aip,
                                      double &out_bip) {
  std::vector<int> pickid;
  for (size_t i = 0; i < oi.size(); i++) {
    pickid.push_back(i);
  }

  if (oi.size() < 4) {
    BOOST_LOG_TRIVIAL(info) << oi.size() << " :Not enough Points for RANSAC\n";
    return 0;
  }
  std::random_device rd;
  std::mt19937 g(rd());

  std::vector<double> found_aips;
  std::vector<double> found_bips;
  int most_inliers = 0;
  std::vector<int> best_inliers;
  std::vector<int> best_outliers;
  for (size_t rsi = 0; rsi < oi.size(); rsi++) {
    std::shuffle(pickid.begin(), pickid.end(), g);
    std::vector<float> this_o, this_op;
    this_o.push_back(oi[pickid[0]]);
    this_o.push_back(oi[pickid[1]]);
    this_o.push_back(oi[pickid[2]]);
    this_o.push_back(oi[pickid[3]]);
    this_op.push_back(opip[pickid[0]]);
    this_op.push_back(opip[pickid[1]]);
    this_op.push_back(opip[pickid[2]]);
    this_op.push_back(opip[pickid[3]]);
    std::vector<double> this_a(1), this_b(1);
    this_a[0] = 1.0;
    this_b[0] = 0.0;

    ceres::Problem problem;
    ceres::CostFunction *cost_function =
        new newSinglePairGainAnaJCostFunc(this_o, this_op);
    problem.AddResidualBlock(cost_function, nullptr, &this_a[0], &this_b[0]);

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Check RANSAC Votes with threshold
    double aip = this_a[0];
    double bip = this_b[0];
    std::vector<int> inliers, outliers;
    double threshold = 8.0e-3;
    for (size_t i = 0; i < oi.size(); i++) {
      double diff = fabs(double(oi[i]) - (double(opip[i]) * (aip - bip) + bip));
      if (diff < threshold) {
        inliers.push_back(i);
      } else {
        outliers.push_back(i);
      }
    }
    found_aips.push_back(aip);
    found_bips.push_back(bip);

    if (inliers.size() > most_inliers) {
      most_inliers = static_cast<int>(inliers.size());
      best_inliers = inliers;
    }
  }

  // Estimate parameters based on inliers
  std::vector<float> inliers_o, inliers_op;
  for (size_t i = 0; i < most_inliers; i++) {
    inliers_o.push_back(oi[best_inliers[i]]);
    inliers_op.push_back(opip[best_inliers[i]]);
  }
  std::vector<double> optimization_variables(2);
  optimization_variables[0] = 1.0;
  optimization_variables[1] = 0.0;
  std::vector<double *> parameter_blocks;
  for (double &optimization_variable : optimization_variables) {
    parameter_blocks.push_back(&optimization_variable);
  }
  ceres::Problem problem;
  problem.AddResidualBlock(
      new newDSinglePairGainAnaJCostFunc(inliers_o, inliers_op, most_inliers),
      nullptr, parameter_blocks);
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  out_aip = *parameter_blocks[0];
  out_bip = *parameter_blocks[1];
  return most_inliers;
}

void IRPhotoCalib::EstimateSpatialParameters(int w, int h, int div,
                                             float GP_length_scale,
                                             float GP_sigma_f, float GP_sigma_n,
                                             std::vector<double> SP_vecB,
                                             std::vector<int> sids_history,
                                             std::vector<int> sids_current) {
  BOOST_LOG_TRIVIAL(debug) << "ESTIMATING SP..." << std::endl;
  std::vector<int> serial_to_variable_id((w * h) / (div * div), -1);
  std::vector<int> variable_to_serial_id;
  std::vector<std::pair<int, int>> Aposneg_id;

  int sid_current = 0;
  int sid_history = 0;
  for (size_t i = 0; i < sids_history.size(); i++) {
    sid_current = sids_current[i];
    sid_history = sids_history[i];

    int vidp = -1;
    int vid = -1;
    if (serial_to_variable_id[sid_current] == -1) {
      // This variable is first. Thus add to serial_to_variable_id and
      // variable_to_serial_id
      variable_to_serial_id.push_back(sid_current);
      int thisVid = static_cast<int>(variable_to_serial_id.size()) - 1;
      serial_to_variable_id[sid_current] = thisVid;
      vidp = thisVid;
    } else
      vidp = serial_to_variable_id[sid_current];

    if (serial_to_variable_id[sid_history] == -1) {
      // This variable is first. Thus add to serial_to_variable_id and
      // variable_to_serial_id
      variable_to_serial_id.push_back(sid_history);
      int thisVid = static_cast<int>(variable_to_serial_id.size()) - 1;
      serial_to_variable_id[sid_history] = thisVid;
      vid = thisVid;
    } else
      vid = serial_to_variable_id[sid_history];

    Aposneg_id.emplace_back(vidp, vid);
  }

  Eigen::MatrixXd b(SP_vecB.size(), 1);
  std::vector<T> tripletList;
  tripletList.reserve(SP_vecB.size() * 2);
  for (size_t i = 0; i < SP_vecB.size(); i++) {
    b(i, 0) = SP_vecB[i];
    tripletList.emplace_back(i, Aposneg_id[i].first, 1.0);
    tripletList.emplace_back(i, Aposneg_id[i].second, -1.0);
  }
  SpMat A(static_cast<long>(SP_vecB.size()),
          static_cast<long>(variable_to_serial_id.size()));
  A.setFromTriplets(tripletList.begin(), tripletList.end());

  // Eigen::SparseQR <SpMat, Eigen::COLAMDOrdering<int> > solver;
  Eigen::LeastSquaresConjugateGradient<SpMat> solver;

  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    BOOST_LOG_TRIVIAL(debug)
        << "problem decomposition failed. Will try again..." << std::endl;
    calibrate_SP_ = true;
    return;
  }
  Eigen::VectorXd x = solver.solve(b);
  if (solver.info() != Eigen::Success) {
    // solving failed
    BOOST_LOG_TRIVIAL(debug)
        << "failed to solve the linear problem. Will try again..." << std::endl;
    calibrate_SP_ = true;
    return;
  }
  GaussianProcessRegression<float> PS_GPR(2, 1);
  PS_GPR.SetHyperParams(GP_length_scale, GP_sigma_f, GP_sigma_n);
  Eigen::VectorXf train_input(2);
  Eigen::VectorXf train_output(1);
  cv::Mat coarse_params_PS(cv::Size((int)(w / div), (int)(h / div)), CV_32FC1,
                           cv::Scalar(0));
  for (size_t i = 0; i < variable_to_serial_id.size(); i++) {
    int sid = variable_to_serial_id[i];
    auto xy = getInvNid(sid);
    train_input << (float)xy.first, (float)xy.second;
    train_output << (float)x(i);
    PS_GPR.AddTrainingData(train_input, train_output);
  }
  for (int c = 0; c < coarse_params_PS.cols; c++) {
    for (int r = 0; r < coarse_params_PS.rows; r++) {
      train_input << (float)c, (float)r;
      train_output = PS_GPR.DoRegression(train_input);
      coarse_params_PS.at<float>(r, c) = train_output(0);
    }
  }

  mutex_.lock();
  for (int c = 0; c < w; c++) {
    for (int r = 0; r < h; r++) {
      params_PS_.at<float>(r, c) =
          coarse_params_PS.at<float>((int)(r / div), (int)(c / div));
    }
  }
  mutex_.unlock();

  BOOST_LOG_TRIVIAL(info) << "Spatial Params Estimation Done!" << std::endl;

  sp_calibrated_ = true;
}

void IRPhotoCalib::getCorrectedImage(cv::Mat image, PTAB &PT_params) {
  cv::Mat float_image, corrected_frame, colormap_corrected_frame;

  const float scale = 1.f / 255.f;
  image.convertTo(float_image, CV_32FC1, scale);
  mutex_.lock();
  cv::Mat corrected_float_frame =
      ((float_image * (float)(PT_params.a - PT_params.b) + (float)PT_params.b) -
       params_PS_) *
      (float)255.0;
  mutex_.unlock();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eigen_image = mapCV2Eigen(corrected_float_frame);
  auto cyclic_eigen_image =
      eigen_image.unaryExpr([](const int x) { return x % 256; }).cast<float>();
  cv::Mat cyclic_float_image = mapEigen2CV(cyclic_eigen_image);
  cyclic_float_image.convertTo(corrected_frame, CV_8UC1, 1);
  cv::LUT(corrected_frame, lut_, image);
}

void IRPhotoCalib::getCorrectedImage(cv::Mat image) {
  cv::Mat float_image, corrected_frame, colormap_corrected_frame,
      corrected_float_frame;
  PTAB &PT_params = params_PT_.back();
  const float scale = 1.f / 255.f;
  image.convertTo(float_image, CV_32FC1, scale);
#ifdef TIMING
  // Set up and start internal timing
  clock_t clock1, clock2, clock3, clock4, clock5;
  clock1 = clock();
#endif
  corrected_float_frame =
      ((float_image * static_cast<float>(PT_params.a - PT_params.b) +
        static_cast<float>(PT_params.b)) -
       params_PS_) *
      255.f;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eigen_image = mapCV2Eigen(corrected_float_frame);
  auto cyclic_eigen_image =
      eigen_image.unaryExpr([](const int x) { return x % 256; }).cast<float>();
  cv::Mat cyclic_float_image = mapEigen2CV(cyclic_eigen_image);
  cyclic_float_image.convertTo(corrected_frame, CV_8UC1, 1);
  cv::LUT(corrected_frame, lut_, image);
#ifdef TIMING
  clock2 = clock();
  BOOST_LOG_TRIVIAL(info)
      << "IMAGE GENERATION  ======================================= "
      << (double)(clock2 - clock1) / CLOCKS_PER_SEC * 1000 << " ms"
      << std::endl;
#endif
}

/*
Source:
https://stackoverflow.com/questions/14783329/opencv-cvmat-and-eigenmatrix/21706778#21706778
You can map arbitrary matrices between Eigen and OpenCV (without copying data).

You have to be aware of two things though:

 - Eigen defaults to column-major storage, OpenCV stores row-major. Therefore,
use the Eigen::RowMajor flag when mapping OpenCV data.
 - The OpenCV matrix has to be continuous (i.e. ocvMatrix.isContinuous() needs
to be true). This is the case if you allocate the storage for the matrix in one
go at the creation of the matrix (e.g. as in Mat A(20, 20, CV_32FC1), or if the
matrix is the result of a operation like Mat W = A.inv();)

For multi-channel matrices (e.g. images), you can use 'Stride' exactly as
Pierluigi suggested in his comment!
** Only works with 32FC1!!**
*/
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
IRPhotoCalib::mapCV2Eigen(cv::Mat &M_OCV) {
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      M_Eigen(M_OCV.ptr<float>(), M_OCV.rows, M_OCV.cols);
  return M_Eigen;
}

template <typename Derived>
cv::Mat IRPhotoCalib::mapEigen2CV(const Eigen::MatrixBase<Derived> &M_D_Eigen) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      M_Eigen = M_D_Eigen;
  cv::Mat M_OCV(M_Eigen.rows(), M_Eigen.cols(), CV_32FC1, M_Eigen.data());
  return M_OCV;
}

bool IRPhotoCalib::isReady() {
  return !params_PT_.empty();  // sp_calibrated_;
}