#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "form/classic/feature.hpp"

namespace form::classic {

using OptionalJacobian = gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

struct ClassicConstraint {
  enum MatchType {
    Plane = 0, // Plane to plane match
    Edge = 1,  // Point to point match
  };

  gtsam::Vector3 n_i;
  gtsam::Point3 p_i;
  gtsam::Point3 p_j;
  MatchType kind;

  inline MatchType parse_match(const PointXYZNTS<double> &p_i,
                               const PointXYZNTS<double> &p_j) noexcept {
    if (p_i.kind == FeatureType::Planar || p_j.kind == FeatureType::Planar) {
      return MatchType::Plane;
    } else {
      return MatchType::Edge;
    }
  }

  ClassicConstraint(const PointXYZNTS<double> &p_i,
                    const PointXYZNTS<double> &p_j) noexcept
      : p_i(p_i.vec3()), p_j(p_j.vec3()), n_i(p_i.n_vec3()),
        kind(parse_match(p_i, p_j)) {}

  ClassicConstraint(const gtsam::Vector3 &p_i_, const gtsam::Vector3 &n_i_,
                    const gtsam::Vector3 &p_j_, const MatchType kind_) noexcept
      : p_i(p_i_), n_i(n_i_), p_j(p_j_), kind(kind_) {}
};

struct PlanePoint {
  Eigen::Matrix3Xd p_i;
  Eigen::Matrix3Xd n_i;
  Eigen::Matrix3Xd p_j;

  PlanePoint() = default;

  PlanePoint(Eigen::Matrix3Xd p_i_, Eigen::Matrix3Xd n_i_,
             Eigen::Matrix3Xd p_j_) noexcept
      : p_i(std::move(p_i_)), n_i(std::move(n_i_)), p_j(std::move(p_j_)) {}

  size_t num_residuals() const noexcept { return p_i.cols(); }
  size_t num_constraints() const noexcept { return p_i.cols(); }

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const noexcept;
};

struct EdgePoint {
  Eigen::Matrix3Xd p_i;
  Eigen::Matrix3Xd d_i;
  Eigen::Matrix3Xd p_j;

  EdgePoint() = default;

  size_t num_residuals() const noexcept { return 3 * p_i.cols(); }
  size_t num_constraints() const noexcept { return p_i.cols(); }

  EdgePoint(Eigen::Matrix3Xd p_i_, Eigen::Matrix3Xd d_i_,
            Eigen::Matrix3Xd p_j_) noexcept
      : p_i(std::move(p_i_)), d_i(std::move(d_i_)), p_j(std::move(p_j_)) {}

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const noexcept;
};

class ClassicFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
public:
  PlanePoint plane;
  EdgePoint edge;

public:
  ClassicFactor(const gtsam::Key i, const gtsam::Key j,
                const std::vector<ClassicConstraint> &constraint,
                double sigma) noexcept;

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
                boost::optional<gtsam::Matrix &> residual_D_Tj =
                    boost::none) const noexcept override;

  [[nodiscard]] gtsam::Key getKey_i() const noexcept { return keys_[0]; }

  [[nodiscard]] gtsam::Key getKey_j() const noexcept { return keys_[1]; }
};

} // namespace form::classic