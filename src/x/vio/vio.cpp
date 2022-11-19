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

#include "x/vio/vio.h"

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <iostream>

#include "x/vio/tools.h"
#include "x/vision/types.h"

// if Boost was compiled with BOOST_NO_EXCEPTIONS defined, it expects a user
// defined trow_exception function, so define a dummy here, if this is the case
#include <exception>

namespace boost {
#ifdef BOOST_NO_EXCEPTIONS
void throw_exception(std::exception const &e){};  // user defined
#endif
}  // namespace boost

using namespace x;
namespace logging = boost::log;

VIO::VIO() : ekf_{Ekf(vio_updater_)} {
  logging::core::get()->set_filter(
#ifdef DEBUG
      logging::trivial::severity >= logging::trivial::debug
#elif VERBOSE
      logging::trivial::severity >= logging::trivial::info
#else
      logging::trivial::severity >= logging::trivial::warning
#endif
  );
}

bool VIO::isInitialized() const { return initialized_; }

void VIO::initAtTime(const double &time) {
  ekf_.lock();
  initialized_ = false;
  initialize_start_ = self_init_start_;
  vio_updater_.track_manager_.clear();
  vio_updater_.state_manager_.clear();

  // Initial IMU measurement (specific force, velocity)
  // Assumption: gravity reaction along IMU +Z axis
  const Vector3 a_m = -params_.g;
  const Vector3 w_m(0.0, 0.0, 0.0);

  //////////////////////////////// xEKF INIT ///////////////////////////////////

  // Initial vision state estimates and uncertainties are all zero
  const int n_poses_state = params_.n_poses_max;
  const int n_features_state = params_.n_slam_features_max;
  const Matrix p_array = Matrix::Zero(n_poses_state * 3, 1);
  const Matrix q_array = Matrix::Zero(n_poses_state * 4, 1);
  const Matrix f_array = Matrix::Zero(n_features_state * 3, 1);
  const Eigen::VectorXd sigma_p_array =
      Eigen::VectorXd::Zero(n_poses_state * 3);
  const Eigen::VectorXd sigma_q_array =
      Eigen::VectorXd::Zero(n_poses_state * 3);
  const Eigen::VectorXd sigma_f_array =
      Eigen::VectorXd::Zero(n_features_state * 3);

  // Construct initial covariance matrix
  const size_t n_err = kSizeCoreErr + n_poses_state * 6 + n_features_state * 3;
  Eigen::VectorXd sigma_diag(n_err);
  sigma_diag << params_.sigma_dp, params_.sigma_dv,
      params_.sigma_dtheta * M_PI / 180.0, params_.sigma_dbw * M_PI / 180.0,
      params_.sigma_dba, sigma_p_array, sigma_q_array, sigma_f_array;

  const Eigen::VectorXd cov_diag = sigma_diag.array() * sigma_diag.array();
  const Matrix cov = cov_diag.asDiagonal();

  // Construct initial state
  const unsigned int dummy_seq = 0;
  State init_state(time, dummy_seq, params_.p, params_.v, params_.q,
                   params_.b_w, params_.b_a, p_array, q_array, f_array, cov,
                   params_.q_ic, params_.p_ic, w_m, a_m);

  // Try to initialize the filter with initial state input

  try {
    ekf_.initializeFromState(init_state);
  } catch (std::runtime_error &e) {
    std::cerr << "bad input: " << e.what() << std::endl;
  } catch (init_bfr_mismatch &e) {
    std::cerr << "init_bfr_mismatch: the size of dynamic arrays in the "
                 "initialization state match must match the size allocated in "
                 "the buffered states."
              << std::endl;
  }
  ekf_.unlock();
  initialized_ = true;
}

void VIO::setUp(const Params &params) {
  // Copy parameters
  params_ = params;

  self_init_start_ = params_.self_init_start_;
  initialize_start_ = self_init_start_;

  // Initialize camera geometry
  camera_ =
      Camera(params_.cam_fx, params_.cam_fy, params_.cam_cx, params_.cam_cy,
             params_.cam_s, params_.img_width, params_.img_height);

  if (params_.min_track_length > params_.n_poses_max) {
    throw std::invalid_argument(
        "'min_track_length' cannot be larger than 'n_poses_max'");
  }

#ifdef MULTI_UAV
  // Set up place recognition module.
  place_recognition_ = std::make_shared<PlaceRecognition>(
      camera_, params_.descriptor_patch_size, params_.descriptor_scale_factor,
      params_.descriptor_pyramid, params_.vocabulary_path,
      params_.fast_detection_delta, params_.desc_type, params_.pr_score_thr,
      params_.pr_desc_min_distance, params_.pr_desc_ratio_thr);
#endif

  // Set up tracker module
  tracker_ =
      Tracker(camera_, params_.fast_detection_delta, params_.non_max_supp,
              params_.block_half_length, params_.margin, params_.n_feat_min,
              params_.outlier_method, params_.outlier_param1,
              params_.outlier_param2, params_.win_size_w, params_.win_size_h,
              params_.max_level, params_.min_eig_thr
#ifdef MULTI_UAV
              ,
              place_recognition_
#endif
#ifdef PHOTOMETRIC_CALI
              ,
              params_.temporal_params_div, params_.spatial_params,
              params_.spatial_params_thr, params_.epsilon_gap,
              params_.epsilon_base, params_.max_level_photo,
              params_.min_eig_thr_photo, params_.win_size_w_photo,
              params_.win_size_h_photo, params_.fast_detection_delta_photo
#endif
      );

  // Compute minimum MSCKF baseline in normal plane (square-pixel assumption
  // along x and y
  msckf_baseline_x_n_ =
      params_.msckf_baseline / (params_.img_width * params_.cam_fx);
  msckf_baseline_y_n_ =
      params_.msckf_baseline / (params_.img_height * params_.cam_fy);

  // Set up tracker and track manager
  track_manager_ =
      TrackManager(camera_, msckf_baseline_x_n_, msckf_baseline_y_n_);

  // Set up VIO state manager
  const int n_poses_state = params_.n_poses_max;
  const int n_features_state = params_.n_slam_features_max;
  const StateManager state_manager(n_poses_state, n_features_state);
  state_manager_ = state_manager;

  // Gravity
  const Vector3 g = params_.g;

  // IMU noise
  ImuNoise imu_noise;
  imu_noise.n_w = params_.n_w;
  imu_noise.n_bw = params_.n_bw;
  imu_noise.n_a = params_.n_a;
  imu_noise.n_ba = params_.n_ba;

  // Updater setup
  MatchList matches;  // unused empty match list since it's image callback
  TiledImage img;
  double sigma_landmark = 0.0;
  double ci_msckf_w = -1.0;
  double ci_slam_w = -1.0;

#ifdef MULTI_UAV
  // init parameters for MULTI_UAV CI-EKF updates
  sigma_landmark = params_.sigma_landmark;
  ci_msckf_w = params_.ci_msckf_w;
  ci_slam_w = params_.ci_slam_w;
#endif

  vio_updater_ =
      VioUpdater(tracker_, state_manager_, track_manager_, params_.sigma_img,
                 params_.sigma_range, params_.rho_0, params_.sigma_rho_0,
                 params_.min_track_length, sigma_landmark, ci_msckf_w,
                 ci_slam_w, params_.iekf_iter);

  // EKF setup
  const size_t state_buffer_sz = 250;  // TODO(jeff) Read from params
  const State default_state = State(n_poses_state, n_features_state);
  const double a_m_max = 50.0;
  const unsigned int delta_seq_imu = 1;
  const auto time_margin_bfr = 0.02;
  ekf_.set(vio_updater_, g, imu_noise, state_buffer_sz, default_state, a_m_max,
           delta_seq_imu, time_margin_bfr);
}

void VIO::setLastRangeMeasurement(const RangeMeasurement &range_measurement) {
  last_range_measurement_ = range_measurement;
}

void VIO::setLastSunAngleMeasurement(
    const SunAngleMeasurement &angle_measurement) {
  last_angle_measurement_ = angle_measurement;
}

std::optional<State> VIO::processImageMeasurement(const double &timestamp,
                                                  const unsigned int seq,
                                                  TiledImage &match_img,
                                                  TiledImage &feature_img) {
#if defined(MULTI_UAV) && defined(DEBUG)
  // used to create the match image in debug mode for the MULTI_UAV system
  matches_img_ = match_img.clone();
#endif
  // Time correction
  const auto timestamp_corrected = timestamp + params_.time_offset;

  BOOST_LOG_TRIVIAL(info) << "Timestamp: " << timestamp << std::endl
                          << "Offset: " << params_.time_offset << std::endl
                          << "Timestamp corrected: " << timestamp_corrected
                          << std::endl;
  // Pass measurement data to updater
  MatchList empty_list;  // TODO(jeff) get rid of image callback and process
  // match list from a separate tracker module.
  VioMeasurement measurement(timestamp_corrected, seq, empty_list, match_img,
                             last_range_measurement_, last_angle_measurement_);

  vio_updater_.setMeasurement(measurement);

// Avoid data race between the Visual update and the Multi UAV update.
// The visual update rewrites and modifies the tracks.
// The multi UAV update reads the tracks.
#if defined(MULTI_THREAD) && defined(MULTI_UAV)
  std::lock_guard<std::mutex> lock(mtx_);
#endif

  // Process update measurement with xEKF
  auto updated_state = ekf_.processUpdateMeasurement();

  // Set state timestamp to original image timestamp for ID purposes in output.
  // We don't do that if that state is invalid, since the timestamp also carries
  // the invalid signature.
  if (updated_state.has_value()) {
    updated_state->setTime(timestamp);
  }

  // Populate GUI image outputs
  match_img = vio_updater_.getMatchImage();
  feature_img = vio_updater_.getFeatureImage();

  return updated_state;
}


std::optional<State> VIO::processMatchesMeasurement(
    const double &timestamp, const unsigned int seq,
    const std::vector<double> &match_vector, TiledImage &match_img,
    TiledImage &feature_img) {
  // Time correction
  const auto timestamp_corrected = timestamp + params_.time_offset;

  // Import matches (except for first measurement, since the previous needs to
  // enter the sliding window)
  MatchList matches;
  if (vio_updater_.state_manager_.poseSize()) {
    matches = importMatches(match_vector, seq, match_img);
  }

  // Compute 2D image coordinates of the LRF impact point on the ground
  Feature lrf_img_pt;
  lrf_img_pt.setXDist(static_cast<double>((camera_.getWidth() + 1) / 2.0));
  lrf_img_pt.setYDist(static_cast<double>((camera_.getHeight() + 1) / 2.0));
  camera_.undistort(lrf_img_pt);
  last_range_measurement_.img_pt = lrf_img_pt;
  last_range_measurement_.img_pt_n = camera_.normalize(lrf_img_pt);

  // Pass measurement data to updater
  VioMeasurement measurement(timestamp_corrected, seq, matches, feature_img,
                             last_range_measurement_, last_angle_measurement_);

  vio_updater_.setMeasurement(measurement);

  // Avoid data race between the Visual update and the Multi UAV update.
  // The visual update rewrites and modifies the tracks.
  // The multi UAV update reads the tracks.
#if defined(MULTI_THREAD) && defined(MULTI_UAV)
  std::lock_guard<std::mutex> lock(mtx_);
#endif

  // Process update measurement with xEKF
  auto updated_state = ekf_.processUpdateMeasurement();

  // Set state timestamp to original image timestamp for ID purposes in output.
  // We don't do that if that state is invalid, since the timestamp carries the
  // invalid signature.
  if (updated_state.has_value()) {
    updated_state->setTime(timestamp);
  }

  // Populate GUI image outputs
  feature_img = vio_updater_.getFeatureImage();

  return updated_state;
}

/** Calls the state manager to compute the cartesian coordinates of the SLAM
 * features.
 */
std::vector<Eigen::Vector3d> VIO::computeSLAMCartesianFeaturesForState(
    const State &state) {
  return vio_updater_.state_manager_.computeSLAMCartesianFeaturesForState(
      state);
}

/** \brief Gets 3D coordinates of MSCKF inliers and outliers.
 *
 *  These are computed in the Measurement class instance.
 */
void VIO::getMsckfFeatures(Vector3dArray &inliers, Vector3dArray &outliers) {
  inliers = vio_updater_.getMsckfInliers();
  outliers = vio_updater_.getMsckfOutliers();
}

std::optional<State> VIO::processImu(const double &timestamp,
                                     const unsigned int seq, const Vector3 &w_m,
                                     const Vector3 &a_m) {

  if (initialize_start_) {
    if (imu_data_batch_.size() < 50) {
        imu_data_batch_.push_back(a_m);
        return std::nullopt;
    }
    imu_data_batch_.push_back(a_m);
    Vector3 avg_a = Vector3::Zero();
    for (const auto &v: imu_data_batch_) {
        avg_a += v;
    }
    avg_a /= static_cast<double>(imu_data_batch_.size());

    Vector3 g(0, 0, a_m.norm());
    params_.q = Quaternion().setFromTwoVectors(avg_a, g);
    // params_.p = Vector3::Zero();
    BOOST_LOG_TRIVIAL(info) << "Initial pose set to: " << params_.q.toRotationMatrix() << std::endl
                            << (params_.q.toRotationMatrix() * a_m);
    initAtTime(timestamp);
    imu_data_batch_.clear();
    initialize_start_ = false;
    return std::nullopt;
  }
  return ekf_.processImu(timestamp, seq, w_m, a_m);
}

MatchList VIO::importMatches(const std::vector<double> &match_vector,
                             const unsigned int seq,
                             TiledImage &img_plot) const {
  // 9N match vector structure:
  // 0: cam_id
  // 1: time_prev in seconds
  // 2: x_dist_prev
  // 3: y_dist_prev
  // 4: time_curr
  // 5: x_dist_curr
  // 6: x_dist_curr
  // 7,8,9: 3D coordinate of feature

  // Number of matches in the input vector
  const unsigned int feature_arr_blk_sz =
      10;  // Length of a feature block in match vector
  assert(match_vector.size() % feature_arr_blk_sz == 0);
  const unsigned int n_matches = match_vector.size() / feature_arr_blk_sz;

  // Declare new lists
  MatchList matches(n_matches);

  // Store the match vector into a match list
  for (unsigned int i = 0; i < n_matches; ++i) {
    // Undistortion
    const double x_dist_prev = match_vector[feature_arr_blk_sz * i + 2];
    const double y_dist_prev = match_vector[feature_arr_blk_sz * i + 3];

    const double x_dist_curr = match_vector[feature_arr_blk_sz * i + 5];
    const double y_dist_curr = match_vector[feature_arr_blk_sz * i + 6];

    // Features and match initializations
    Feature previous_feature(match_vector[feature_arr_blk_sz * i + 1], seq - 1,
                             0.0, 0.0, x_dist_prev, y_dist_prev, -1.0);
    camera_.undistort(previous_feature);

    Feature current_feature(match_vector[feature_arr_blk_sz * i + 4], seq, 0.0,
                            0.0, x_dist_curr, y_dist_curr, -1.0);
    camera_.undistort(current_feature);

#ifdef GT_DEBUG
    // 3D landmark given by the ground truth
    const double x_l = match_vector[feature_arr_blk_sz * i + 7];
    const double y_l = match_vector[feature_arr_blk_sz * i + 8];
    const double z_l = match_vector[feature_arr_blk_sz * i + 9];
    Vector3 landmark = Vector3(x_l, y_l, z_l);
    previous_feature.setLandmark(landmark);
    current_feature.setLandmark(landmark);
#endif

    Match current_match;
    current_match.previous = previous_feature;
    current_match.current = current_feature;

    // Add match to list
    matches[i] = current_match;
  }

  // Publish matches to GUI
  Tracker::plotMatches(matches, img_plot);

  return matches;
}

#ifdef MULTI_UAV

#ifndef REQUEST_COMM

void VIO::getDataToSend(std::shared_ptr<SimpleState> &state_ptr,
                        const State &state, TrackList &msckf_tracks,
                        TrackList &slam_tracks, std::vector<int> &anchor_idxs,
                        TrackList &opp_tracks) {
  vio_updater_.getMsckfTracks(msckf_tracks);
  vio_updater_.getSlamTracks(slam_tracks, anchor_idxs, state.nPosesMax());
  vio_updater_.getOppTracks(opp_tracks);
  state_ptr = std::make_shared<SimpleState>(
      state.getDynamicStates(), state.getPositionArray(),
      state.getOrientationArray(), state.getFeatureArray(),
      state.getCovariance(), anchor_idxs);
}

#endif

cv::Mat VIO::getDescriptors() {
#ifdef MULTI_THREAD
  std::lock_guard<std::mutex> lock(mtx_);
#endif
  return place_recognition_->getDescriptors();
}

void VIO::processOtherRequests(const int uav_id, cv::Mat &descriptors,
                               std::shared_ptr<SimpleState> &state,
                               TrackList &msckf_tracks, TrackList &slam_tracks,
                               std::vector<int> &anchor_idxs,
                               TrackList &opp_tracks) {
#ifdef TIMING
  // Set up and start internal timing
  clock_t clock1, clock2;
  clock1 = clock();
#endif

  KeyframePtr candidate;  // the candidate will be sent to the UAV `uav_id` as
  // measurement message
  place_recognition_->findPlace(uav_id, descriptors, candidate);
#ifdef TIMING
  clock2 = clock();
  BOOST_LOG_TRIVIAL(info) << "Process Other Requests ======================="
                          << std::endl;
  BOOST_LOG_TRIVIAL(info) << "Times ===================================="
                          << std::endl
                          << "Place recognition:               "
                          << (double)(clock2 - clock1) / CLOCKS_PER_SEC * 1000
                          << " ms" << std::endl
                          << "=========================================="
                          << std::endl;
#endif

  if (candidate != nullptr) {
    state = candidate->getState();
    slam_tracks = candidate->getSlamTracks();
    msckf_tracks = candidate->getMsckfTracks();
    opp_tracks = candidate->getOppTracks();
    anchor_idxs = candidate->getAnchors();
  }
}

std::optional<State> VIO::processOtherMeasurements(
    double timestamp, const int uav_id, const Vectorx &dynamic_state,
    const Vectorx &positions_state, const Vectorx &orientations_state,
    const Vectorx &features_state, const Matrix &cov,
    const TrackListPtr &received_msckf_trcks_ptr,
    const TrackListPtr &received_slam_trcks_ptr,
    const TrackListPtr &received_opp_tracks_ptr,
    const std::vector<int> &anchor_idxs, cv::Mat &dst) {
  std::shared_ptr<SimpleState> ptr = std::make_shared<SimpleState>(
      dynamic_state, positions_state, orientations_state, features_state, cov,
      anchor_idxs);

/** Avoid data race between the Visual update and the Multi UAV update.
 *  The visual update rewrites and modifies the tracks.
 *  The multi UAV update reads the tracks.
 * */
#ifdef MULTI_THREAD
  std::lock_guard<std::mutex> lock(mtx_);
#endif
  // retrieve current UAV data
  TrackList current_msckf, current_slam, current_opp_tracks;
  vio_updater_.getMsckfTracks(current_msckf);
  vio_updater_.getSlamTracks(current_slam);
  vio_updater_.getOppTracks(current_opp_tracks);
  int corr_idx = 0;
  for (size_t i = 0; i < current_opp_tracks.size(); i++) {
    if (current_opp_tracks[i - corr_idx].size() < 3) {
      current_opp_tracks.erase(current_opp_tracks.begin() + i - corr_idx);
      corr_idx++;
    }
  }

  // Proceede iif there are tracks.
  if (current_slam.empty() && current_opp_tracks.empty()) {
    return std::nullopt;
  }

#ifdef TIMING
  // Set up and start internal timing
  clock_t clock1, clock2;
  clock1 = clock();
#endif

  // Find place correspondence.
  bool place_found = place_recognition_->findCorrespondences(
      uav_id, current_msckf, received_msckf_trcks_ptr, current_slam,
      received_slam_trcks_ptr, current_opp_tracks, received_opp_tracks_ptr, ptr,
      dst, dst);

  if (!place_found) {
    return std::nullopt;
  } else {
#ifdef DEBUG
    BOOST_LOG_TRIVIAL(info)  << "\033[1;33mPlace recognition\033[0m found correspondence!"
              << std::endl;
#endif
  }
  // process the SLAM updates
  auto updated_state = ekf_.processOthersMeasurement(timestamp);

#ifdef TIMING
  clock2 = clock();
  BOOST_LOG_TRIVIAL(info)  << "Process Other Measurements ======================="
            << std::endl;
  BOOST_LOG_TRIVIAL(info)  << "Times ====================================" << std::endl
            << "Find matches:               "
            << (double)(clock2 - clock1) / CLOCKS_PER_SEC * 1000 << " ms"
            << std::endl
            << "==========================================" << std::endl;
#endif

  return updated_state;
}


#endif


Params loadParamsFromYaml(fsm::path& filePath) {
  std::cout << filePath.string() << std::endl;
  cv::FileStorage file(filePath.string(), cv::FileStorage::READ);
  Params params;
  std::vector<double> p, v, q, b_w, b_a, p_ic, q_ic, q_sc, w_s, sigma_dp,
      sigma_dv, sigma_dtheta, sigma_dbw, sigma_dba, g;

  file["p"] >> p;
  file["v"] >> v;
  file["q"] >> q;
  file["b_w"] >> b_w;
  file["b_a"] >> b_a;

  params.p << p[0], p[1], p[2];
  params.v << v[0], v[1], v[2];
  params.q.w() = q[0];
  params.q.x() = q[1];
  params.q.y() = q[2];
  params.q.z() = q[3];
  params.q.normalize();
  params.b_w << b_w[0], b_w[1], b_w[2];
  params.b_a << b_a[0], b_a[1], b_a[2];

  file["sigma_dp"] >> sigma_dp;
  file["sigma_dv"] >> sigma_dv;
  file["sigma_dtheta"] >> sigma_dtheta;
  file["sigma_dbw"] >> sigma_dbw;
  file["sigma_dba"] >> sigma_dba;

  params.sigma_dp << sigma_dp[0], sigma_dp[1], sigma_dp[2];
  params.sigma_dv << sigma_dv[0], sigma_dv[1], sigma_dv[2];
  params.sigma_dtheta << sigma_dtheta[0], sigma_dtheta[1], sigma_dtheta[2];
  params.sigma_dbw << sigma_dbw[0], sigma_dbw[1], sigma_dbw[2];
  params.sigma_dba << sigma_dba[0], sigma_dba[1], sigma_dba[2];

  file["cam1_fx"] >> params.cam_fx;
  file["cam1_fy"] >> params.cam_fy;
  file["cam1_cx"] >> params.cam_cx;
  file["cam1_cy"] >> params.cam_cy;
  file["cam1_s"] >> params.cam_s;

  file["cam1_img_height"] >> params.img_height;
  file["cam1_img_width"] >> params.img_width;
  file["cam1_p_ic"] >> p_ic;
  file["cam1_q_ic"] >> q_ic;

  params.p_ic << p_ic[0], p_ic[1], p_ic[2];
  params.q_ic.w() = q_ic[0];
  params.q_ic.x() = q_ic[1];
  params.q_ic.y() = q_ic[2];
  params.q_ic.z() = q_ic[3];
  params.q_ic.normalize();

  file["cam1_time_offset"] >> params.time_offset;
  file["sigma_img"] >> params.sigma_img;

  file["sigma_range"] >> params.sigma_range;
  file["q_sc"] >> q_sc;
  file["w_s"] >> w_s;

  params.q_sc.w() = q_sc[0];
  params.q_sc.x() = q_sc[1];
  params.q_sc.y() = q_sc[2];
  params.q_sc.z() = q_sc[3];
  params.q_sc.normalize();
  params.w_s << w_s[0], w_s[1], w_s[2];
  params.w_s.normalize();

  file["n_a"] >> params.n_a;
  file["n_ba"] >> params.n_ba;
  file["n_w"] >> params.n_w;
  file["n_bw"] >> params.n_bw;

#ifdef PHOTOMETRIC_CALI
  file["temporal_params_div"] >> params.temporal_params_div;
  file["spatial_params"] >> params.spatial_params;
  file["spatial_params_thr"] >> params.spatial_params_thr;
  file["epsilon_gap"] >> params.epsilon_gap;
  file["epsilon_base"] >> params.epsilon_base;
  file["win_size_w_photo"] >> params.win_size_w_photo;
  file["win_size_h_photo"] >> params.win_size_h_photo;
  file["max_level_photo"] >> params.max_level_photo;
  file["min_eig_thr_photo"] >> params.min_eig_thr_photo;
  file["fast_detection_delta_photo"] >> params.fast_detection_delta_photo;
#endif
#ifdef MULTI_UAV
  file["vocabulary_path"] >> params.vocabulary_path;
  file["sigma_landmark"] >> params.sigma_landmark;
  file["descriptor_scale_factor"] >> params.descriptor_scale_factor;
  file["descriptor_pyramid"] >> params.descriptor_pyramid;
  file["descriptor_patch_size"] >> params.descriptor_patch_size;
  file["ci_msckf_w"] >> params.ci_msckf_w;
  file["ci_slam_w"] >> params.ci_slam_w;
  file["desc_type"] >> params.desc_type;
  file["pr_score_thr"] >> params.pr_score_thr;
  file["pr_desc_ratio_thr"] >> params.pr_desc_ratio_thr;
  file["pr_desc_min_distance"] >> params.pr_desc_min_distance;
#endif

  file["min_eig_thr"] >> params.min_eig_thr;
  file["max_level"] >> params.max_level;
  file["win_size_w"] >> params.win_size_w;
  file["win_size_h"] >> params.win_size_h;

  file["fast_detection_delta"] >> params.fast_detection_delta;
  file["non_max_supp"] >> params.non_max_supp;
  file["block_half_length"] >> params.block_half_length;
  file["margin"] >> params.margin;
  file["n_feat_min"] >> params.n_feat_min;
  file["outlier_method"] >> params.outlier_method;
  file["outlier_param1"] >> params.outlier_param1;
  file["outlier_param2"] >> params.outlier_param2;
  file["n_tiles_h"] >> params.n_tiles_h;
  file["n_tiles_w"] >> params.n_tiles_w;
  file["max_feat_per_tile"] >> params.max_feat_per_tile;

  file["n_poses_max"] >> params.n_poses_max;
  file["n_slam_features_max"] >> params.n_slam_features_max;
  file["rho_0"] >> params.rho_0;
  file["sigma_rho_0"] >> params.sigma_rho_0;
  file["iekf_iter"] >> params.iekf_iter;
  file["msckf_baseline"] >> params.msckf_baseline;
  file["min_track_length"] >> params.min_track_length;
  file["state_buffer_size"] >> params.state_buffer_size;


  file["g"] >> g;

  params.g << g[0], g[1], g[2];

  return params;
}