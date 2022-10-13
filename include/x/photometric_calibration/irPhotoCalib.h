/*
 * Author: Manash Pratim Das (mpdmanash@iitkgp.ac.in)
 */

#if !defined(IRPHOTOCALIB_H) && defined(PHOTOMETRIC_CALI)
#define IRPHOTOCALIB_H

//#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "x/photometric_calibration/gaussian_process_regression.h"
#include "x/vio/types.h"

using namespace x;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> T;

struct PTAB {
  double a;
  double b;
};

//#define M_PI 3.14159265

class IRPhotoCalib {
 public:
  IRPhotoCalib(int w, int h, int k_div, bool k_calibrate_SP,
               float k_SP_threshold, double epsilon_gap, double epsilon_base,
               bool use_keyframes = true);
  ~IRPhotoCalib();
  void ProcessCurrentFrame(
      std::vector<std::vector<float>> &intensity_history,
      std::vector<std::vector<float>> &intensity_current,
      const std::vector<int> &frame_ids_history,
      std::vector<std::vector<std::pair<int, int>>> &pixels_history,
      std::vector<std::vector<std::pair<int, int>>> &pixels_current,
      bool thisKF = false);
  [[nodiscard]] int EstimateGainsRansac(std::vector<float> oi,
                                        std::vector<float> oip, double &out_aip,
                                        double &out_bip);
  void getCorrectedImage(cv::Mat image, PTAB &PT_params);
  void getCorrectedImage(cv::Mat image);
  static void getRelativeGains(double a1, double b1, double a2, double b2,
                               double &a12, double &b12);
  static void chainGains(double a01, double b01, double a12, double b12,
                         double &a02, double &b02);

  bool isReady();

 private:
  const bool use_keyframes_;
  bool sp_calibrated_ = false;
  std::vector<PTAB> params_PT_;
  std::vector<int> m_KF_ids;
  const int div_, w_, h_;
  int latest_KF_id_ = 0, frame_id_ = 0;
  const double epsilon_gap_, epsilon_base_;
  float SP_threshold_, GP_length_scale_, GP_sigma_f_, GP_sigma_n_;
  cv::Mat spatial_coverage_, params_PS_, lut_;
  [[nodiscard]] PTAB getPrevAB();
  [[nodiscard]] int getNid(int ptx, int pty) const;
  [[nodiscard]] std::pair<int, int> getInvNid(int sid) const;
  void EstimateSpatialParameters(int w, int h, int div, float GP_length_scale,
                                 float GP_sigma_f, float GP_sigma_n,
                                 std::vector<double> SP_vecB,
                                 std::vector<int> sids_history,
                                 std::vector<int> sids_current);

  template <typename Derived>
  [[nodiscard]] cv::Mat mapEigen2CV(const Eigen::MatrixBase<Derived> &M_Eigen);
  [[nodiscard]] Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>
  mapCV2Eigen(cv::Mat &M_OCV);

  // Spatial Parameters
  std::vector<double> m_SP_vecB;
  std::vector<int> SP_correscount_;
  std::vector<int> sids_history_;
  std::vector<int> sids_current_;
  int SP_max_correscount_;

  std::atomic<bool> calibrate_SP_{false};
  std::mutex photo_mtx_;
  std::thread spatial_par_thread_;

  std::mutex mutex_;
};

#endif
