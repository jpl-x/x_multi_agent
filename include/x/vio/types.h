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

#ifndef X_VIO_TYPES_H
#define X_VIO_TYPES_H

#include <Eigen/Dense>
#include <vector>

#include "x/vision/feature.h"
#include "x/vision/tiled_image.h"
#include "x/vision/types.h"
/**
 * This header defines common types used in xVIO.
 */
namespace x {
/**
 * User-defined parameters
 */
struct Params {
  // position vector
  Vector3 p;
  // velocity vector
  Vector3 v;
  // orientation quaternion
  Quaternion q;
  // biases for acc and gyro
  Vector3 b_w;
  Vector3 b_a;
  // Standard deviation for p, v, q, and biases
  Vector3 sigma_dp;
  Vector3 sigma_dv;
  Vector3 sigma_dtheta;
  Vector3 sigma_dbw;
  Vector3 sigma_dba;
  // camera parameters
  double cam_fx;
  double cam_fy;
  double cam_cx;
  double cam_cy;
  double cam_s;
  int img_height;
  int img_width;
  // imu-camera calibration parameters
  Vector3 p_ic;
  Quaternion q_ic;

  // Standard deviation of feature measurement [in normalized coordinates]
  double sigma_img;

  // Standard deviation of range measurement noise [m]
  double sigma_range;

  // Sun sensor params
  Quaternion q_sc;
  Vector3 w_s;

  // IMU noise and random walk parameters
  double n_a;
  double n_ba;
  double n_w;
  double n_bw;

  // Tracking and detector paramters
  int fast_detection_delta;
  bool non_max_supp;
  int block_half_length;
  int margin;
  int n_feat_min;
  int outlier_method;
  double outlier_param1;
  double outlier_param2;
  int n_tiles_h;
  int n_tiles_w;
  int max_feat_per_tile;
  double time_offset;

#ifdef PHOTOMETRIC_CALI
  int temporal_params_div;
  bool spatial_params;
  double spatial_params_thr;
  double epsilon_gap;
  double epsilon_base;

  // before the calibration we need to track features as the image is not
  // calibrated. After the calibration the parameters need to be changed since
  // the images are then calibrated.
  // KLT parameters
  int max_level_photo;
  double min_eig_thr_photo;
  int win_size_w_photo;
  int win_size_h_photo;
  int fast_detection_delta_photo;
#endif

#ifdef MULTI_UAV
  std::string vocabulary_path;
  double sigma_landmark;

  /**
   * ORB descriptor extractor parameters
   */
  float descriptor_scale_factor;
  int descriptor_pyramid;
  int descriptor_patch_size;

  double ci_msckf_w;
  double ci_slam_w;

  int desc_type;  // enum DESCRIPTOR_TYPE { ORB=0, SIFT, SURF };
  double pr_score_thr;

  double pr_desc_ratio_thr;
  double pr_desc_min_distance;
#endif

  /**
   * KLT parameters
   */
  int max_level;
  double min_eig_thr;
  int win_size_w;
  int win_size_h;

  /**
   * Maximum number of poses in the sliding window.
   */
  int n_poses_max = 15;

  /**
   * Maximum number of SLAM features.
   */
  int n_slam_features_max = 15;

  /**
   * Initial inverse depth of SLAM features [1/m].
   *
   * This is when SLAM features can't be triangulated for MSCKK-SLAM. By
   * default, this should be 1 / (2 * d_min), with d_min the minimum
   * expected feature depth (2-sigma) [Montiel, 2006]
   */
  double rho_0 = 0.5;

  /**
   * Initial standard deviation of SLAM inverse depth [1/m].
   *
   * This is when SLAM features can't be triangulated for MSCKK-SLAM. By
   * default, this should be 1 / (4 * d_min), with d_min the minimum
   * expected feature depth (2-sigma) [Montiel, 2006].
   */
  double sigma_rho_0 = 0.25;

  /**
   * Number of IEKF iterations (EKF <=> 1).
   */
  int iekf_iter = 1;

  /**
   * Minimum baseline to trigger MSCKF measurement (pixels).
   */
  double msckf_baseline = 10;

  /**
   * Minimum track length for a visual feature to be processed.
   */
  int min_track_length = 15;

  /**
   * Gravity vector in world frame [x,y,z]m in m/s^2
   */
  Vector3 g;

  bool self_init_start_ = false;

  int state_buffer_size = 250;
};

/**
 * MSCKF-SLAM matrices
 */
struct MsckfSlamMatrices {
  /**
   * H1 matrix in Li's paper (stacked for all features)
   */
  Matrix H1;

  /**
   * H2 matrix in Li's paper (stacked for all features)
   */
  Matrix H2;

  /**
   * z1 residual vector in Li's paper (stacked for all features)
   */
  Matrix r1;

  /**
   * New SLAM feature vectors (stacked for all features).
   *
   * Features' inverse-depth coordinates, assuming the latest camera as an
   * anchor. These estimates are taken from the MSCKF triangulation prior.
   */
  Matrix features;
};

/**
 * Range measurement.
 *
 * Coming from a Laser Range Finder (LRF).
 */
struct RangeMeasurement {
  double timestamp{kInvalid};

  double range{0.0};

  /**
   * Image coordinate of the LRF beam.
   *
   * This is assumed to be a single point in the image, i.e. the LRF axis
   * passes by the optical center of the camera.
   */
  Feature img_pt{Feature()};

  /**
   * Normalized image coordinates of the LRF beam.
   *
   * Follows the same assumptions as img_pt.
   */
  Feature img_pt_n{Feature()};
};

/**
 * Sun angle measurement.
 *
 * Coming from a sun sensor.
 */
struct SunAngleMeasurement {
  double timestamp{kInvalid};
  double x_angle{0.0};
  double y_angle{0.0};
};

/**
 * VIO update measurement.
 *
 * This struct includes all sensor the measurements that will processed to
 * create an EKF update. This should be at least an image (or a set of visual
 * matches) in xVIO, along side optional additional sensor measurements that
 * are assumed to be synced with the visual data (LRF and sun sensor).
 */
struct VioMeasurement {
  /**
   * Timestamp.
   *
   * This timestamp is the visual measurement timestamp (camera or match
   * list). Range and sun angle measurement timestamps might different but
   * will be processed at the sam timestamp as a single EKF update.
   */
  double timestamp{0};

  /**
   * Sequence ID.
   *
   * Consecutively increasing ID associated with the visual measurement
   * (matches or image).
   */
  unsigned int seq{0};

  /**
   * Visual match measurements.
   *
   * Output of a visual feature tracker (or simulation).
   */
  MatchList matches;

  /**
   * Image measurement.
   *
   * Will only be used if the visual match list struct member has size zero,
   * in which case a feature track will run on that image.
   */
  TiledImage image;

  /**
   * Range measurement.
   */
  RangeMeasurement range;

  /**
   * Sun angle measurement.
   */
  SunAngleMeasurement sun_angle;

  /**
   * Default constructor.
   */
  VioMeasurement() = default;

  /**
   * Full constructor.
   */
  VioMeasurement(const double &timestamp, const unsigned int seq,
                 MatchList matches, const TiledImage &image,
                 RangeMeasurement range, const SunAngleMeasurement &sun_angle)
      : timestamp{timestamp},
        seq{seq},
        matches{std::move(matches)},
        image{image},
        range{std::move(range)},
        sun_angle{sun_angle} {}
};

using Vector3dArray = std::vector<Eigen::Vector3d>;
}  // namespace x

#endif  // X_VIO_TYPES_H
