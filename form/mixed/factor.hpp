#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "form/mixed/feature.hpp"

namespace form::mixed {

using OptionalJacobian = gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

struct MixedConstraint {
  enum MatchType {
    PlanePlane = 0, // Plane to plane match
    PlanePoint = 1, // Plane to point match
    PointPlane = 2, // Point to plane match
    PointPoint = 3, // Point to point match
  };

  gtsam::Vector3 n_i;
  gtsam::Point3 p_i;
  gtsam::Vector3 n_j;
  gtsam::Point3 p_j;
  MatchType type;

  inline MatchType parse_match(const PointXYZNTS<double> &p_i,
                               const PointXYZNTS<double> &p_j) noexcept {
    if (p_i.kind == FeatureType::Planar && p_j.kind == FeatureType::Planar) {
      return MatchType::PlanePlane;
    } else if (p_i.kind == FeatureType::Planar && p_j.kind == FeatureType::Point) {
      return MatchType::PlanePoint;
    } else if (p_i.kind == FeatureType::Point && p_j.kind == FeatureType::Planar) {
      return MatchType::PointPlane;
    } else {
      return MatchType::PointPoint;
    }
  }

  MixedConstraint(const PointXYZNTS<double> &p_i,
                  const PointXYZNTS<double> &p_j) noexcept
      : p_i(p_i.vec3()), p_j(p_j.vec3()), n_i(p_i.n_vec3()), n_j(p_j.n_vec3()),
        type(parse_match(p_i, p_j)) {}

  // Testing constructors
  // point point constructor
  MixedConstraint(const gtsam::Vector3 &p_i, const gtsam::Vector3 &p_j) noexcept
      : p_i(p_i), p_j(p_j), n_i(gtsam::Vector3::Zero()), n_j(gtsam::Vector3::Zero()),
        type(MatchType::PointPoint) {}

  // plane point constructor
  MixedConstraint(const gtsam::Vector3 &p_i, const gtsam::Vector3 &n_i,
                  const gtsam::Vector3 &p_j) noexcept
      : p_i(p_i), p_j(p_j), n_i(n_i), n_j(gtsam::Vector3::Zero()),
        type(MatchType::PlanePoint) {}

  // point plane constructor
  MixedConstraint(const gtsam::Vector3 &p_i, const gtsam::Vector3 &p_j,
                  const gtsam::Vector3 &n_j, bool flag) noexcept
      : p_i(p_i), p_j(p_j), n_i(gtsam::Vector3::Zero()), n_j(n_j),
        type(MatchType::PointPlane) {}

  // plane plane constructor
  MixedConstraint(const gtsam::Vector3 &p_i, const gtsam::Vector3 &n_i,
                  const gtsam::Vector3 &p_j, const gtsam::Vector3 &n_j) noexcept
      : p_i(p_i), p_j(p_j), n_i(n_i), n_j(n_j), type(MatchType::PlanePlane) {}
};

struct PlanePoint {
  Eigen::Matrix3Xd p_i;
  Eigen::Matrix3Xd n_i;
  Eigen::Matrix3Xd p_j;

  PlanePoint() = default;

  size_t size() const noexcept { return p_i.cols(); }

  PlanePoint(Eigen::Matrix3Xd p_i_, Eigen::Matrix3Xd n_i_,
             Eigen::Matrix3Xd p_j_) noexcept
      : p_i(std::move(p_i_)), n_i(std::move(n_i_)), p_j(std::move(p_j_)) {}

  size_t num_constraints() const noexcept { return p_i.cols(); }

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const noexcept;
};

struct PointPoint {
  Eigen::Matrix3Xd p_i;
  Eigen::Matrix3Xd p_j;

  PointPoint() = default;

  size_t size() const noexcept { return 3 * p_i.cols(); }

  size_t num_constraints() const noexcept { return p_i.cols(); }

  PointPoint(Eigen::Matrix3Xd p_i_, Eigen::Matrix3Xd p_j_) noexcept
      : p_i(std::move(p_i_)), p_j(std::move(p_j_)) {}

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const noexcept;
};

struct PlanePlane {
  PlanePoint plane_point;
  PlanePoint point_plane;

  PlanePlane() = default;

  PlanePlane(Eigen::Matrix3Xd p_i_, Eigen::Matrix3Xd n_i_, Eigen::Matrix3Xd p_j_,
             Eigen::Matrix3Xd n_j_) noexcept
      : plane_point(p_i_, n_i_, p_j_), point_plane(p_j_, n_j_, p_i_) {}

  size_t size() const noexcept { return 2 * plane_point.size(); }

  size_t num_constraints() const noexcept { return plane_point.num_constraints(); }

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const noexcept;
};

class MixedFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
public:
  PlanePoint plane_plane;
  PlanePoint plane_point;
  PlanePoint point_plane;
  PointPoint point_point;

public:
  MixedFactor(const gtsam::Key i, const gtsam::Key j,
              const std::vector<MixedConstraint> &constraint, double sigma,
              double weight) noexcept;

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
                boost::optional<gtsam::Matrix &> residual_D_Tj =
                    boost::none) const noexcept override;

  [[nodiscard]] gtsam::Key getKey_i() const noexcept { return keys_[0]; }

  [[nodiscard]] gtsam::Key getKey_j() const noexcept { return keys_[1]; }
};

} // namespace form::mixed