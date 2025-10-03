#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "form/point/feature.hpp"

namespace form::point {

struct PointConstraint {
  gtsam::Point3 p_i;
  gtsam::Point3 p_j;

  PointConstraint(const PointXYZS<double> &p_i,
                  const PointXYZS<double> &p_j) noexcept
      : p_i(p_i.vec3()), p_j(p_j.vec3()) {}

  PointConstraint(const gtsam::Point3 &p_i, const gtsam::Point3 &p_j) noexcept
      : p_i(p_i), p_j(p_j) {}

  [[nodiscard]] gtsam::Vector evaluateError(
      const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
      boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
      boost::optional<gtsam::Matrix &> residual_D_Tj = boost::none) const noexcept;
};

class PointFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {

public:
  size_t size;
  Eigen::Matrix3Xd p_i;
  Eigen::Matrix3Xd p_j;

public:
  PointFactor(const gtsam::Key i, const gtsam::Key j,
              const std::vector<PointConstraint> &constraint, double sigma) noexcept;

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
                boost::optional<gtsam::Matrix &> residual_D_Tj =
                    boost::none) const noexcept override;

  [[nodiscard]] gtsam::Key getKey_i() const noexcept { return keys_[0]; }

  [[nodiscard]] gtsam::Key getKey_j() const noexcept { return keys_[1]; }
};

} // namespace form::point
