//
// Created by viciopoli on 05.01.23.
//

#ifndef X_TYPES_H
#define X_TYPES_H

namespace x::Camera {

    enum DistortionModel {
        FOV, RADTAN, EQUIDISTANT, NONE
    };

    struct Params {
        unsigned int img_width_;
        unsigned int img_height_;

        double fx_;
        double fy_;
        double cx_;
        double cy_;

        std::vector<double> dist_coeff;

        double inv_fx_;
        double inv_fy_;
        double cx_n_;
        double cy_n_;

        DistortionModel cameraType;

        Params(double fx,
               double fy,
               double cx,
               double cy,
               std::vector<double> dist_coeff,
               unsigned int img_width,
               unsigned int img_height,
               std::string distortion_model) :
                img_width_(img_width),
                img_height_(img_height),
                fx_(fx * img_width), fy_(fy * img_height),
                cx_(cx * img_width), cy_(cy * img_height),
                dist_coeff(std::move(dist_coeff)),
                inv_fx_(1.0 / fx_),
                inv_fy_(1.0 / fy_), cx_n_(cx_ * inv_fx_), cy_n_(cy_ * inv_fy_) {
            cameraType = distortionString2Enum(distortion_model);
        };

        DistortionModel distortionString2Enum(const std::string &distortion_model) {
            if (distortion_model == "FOV") {
                return DistortionModel::FOV;
            } else if (distortion_model == "RADTAN") {
                return DistortionModel::RADTAN;
            } else if (distortion_model == "EQUIDISTANT") {
                return DistortionModel::EQUIDISTANT;
            } else {
                return DistortionModel::NONE;
            }
        }


        [[nodiscard]] cv::Matx33d getCameraMatrix() const {
            static cv::Matx33d m(fx_, 0.0, cx_,
                                 0.0, fy_, cy_,
                                 0.0, 0.0, 1.0);
            return m;
        }

        [[nodiscard]] Matrix3 getCameraMatrixEigen() const {
            static Matrix3 m;
            m << fx_, 0.0, cx_,
                    0.0, fy_, cy_,
                    0.0, 0.0, 1.0;
            return m;
        }
    };
}
#endif //X_TYPES_H
