#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "form/planar/feature.hpp"

namespace form::planar {

struct PlanarConstraint {
  gtsam::Vector3 n_i;
  gtsam::Point3 p_i;
  gtsam::Vector3 n_j;
  gtsam::Point3 p_j;

  PlanarConstraint(const PointXYZNS<double> &p_i,
                   const PointXYZNS<double> &p_j) noexcept
      : p_i(p_i.vec3()), p_j(p_j.vec3()), n_i(p_i.n_vec3()), n_j(p_j.n_vec3()) {}

  PlanarConstraint(const gtsam::Point3 &p_i, const gtsam::Vector3 &n_i,
                   const gtsam::Point3 &p_j,
                   const gtsam::Vector3 &n_j = gtsam::Vector3::Zero()) noexcept
      : p_i(p_i), n_i(n_i), p_j(p_j), n_j(n_j) {}

  [[nodiscard]] gtsam::Vector evaluateError(
      const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
      bool use_plane_to_plane = false,
      boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
      boost::optional<gtsam::Matrix &> residual_D_Tj = boost::none) const noexcept;
};

class PlanarFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {

public:
  bool m_use_plane_to_plane;
  size_t size;
  Eigen::Matrix3Xd p_i;
  Eigen::Matrix3Xd p_j;
  Eigen::Matrix3Xd n_i;
  Eigen::Matrix3Xd n_j;

public:
  PlanarFactor(const gtsam::Key i, const gtsam::Key j,
               const std::vector<PlanarConstraint> &constraint,
               double sigma) noexcept;

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
                boost::optional<gtsam::Matrix &> residual_D_Tj =
                    boost::none) const noexcept override;

  [[nodiscard]] gtsam::Key getKey_i() const noexcept { return keys_[0]; }

  [[nodiscard]] gtsam::Key getKey_j() const noexcept { return keys_[1]; }
};

class PlanarFactor1 : public gtsam::NoiseModelFactor1<gtsam::Pose3> {

private:
  bool m_use_plane_to_plane;
  gtsam::Pose3 m_Ti;
  std::vector<PlanarConstraint> m_constraints;

public:
  PlanarFactor1(const gtsam::Pose3 Ti, const gtsam::Key j,
                const std::vector<PlanarConstraint> &constraint, double sigma,
                bool use_plane_to_plane = false) noexcept;

  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Tj,
                boost::optional<gtsam::Matrix &> residual_D_Tj =
                    boost::none) const noexcept override;

  [[nodiscard]] gtsam::Key getKey_i() const noexcept { return keys_[0]; }

  [[nodiscard]] gtsam::Key getKey_j() const noexcept { return keys_[1]; }

  [[nodiscard]] const std::vector<PlanarConstraint> &constraints() const noexcept {
    return m_constraints;
  }
};

} // namespace form::planar