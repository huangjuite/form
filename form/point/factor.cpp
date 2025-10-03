#include "form/point/factor.hpp"
#include <gtsam/base/Matrix.h>

namespace form::point {

// TODO: Probably easier to just split this into two functions
[[nodiscard]] gtsam::Vector
PointConstraint::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                               boost::optional<gtsam::Matrix &> H1,
                               boost::optional<gtsam::Matrix &> H2) const noexcept {
  gtsam::Matrix33 w_ni_D_Ri;
  gtsam::Matrix36 w_pi_D_Ti;
  gtsam::Matrix33 w_nj_D_Rj;
  gtsam::Matrix36 w_pj_D_Tj;
  const double v_D_w_pj = 1.0;
  const double v_D_w_pi = -1.0;

  const gtsam::Point3 w_pi = Ti.transformFrom(p_i, H1 ? &w_pi_D_Ti : 0);
  const gtsam::Point3 w_pj = Tj.transformFrom(p_j, H2 ? &w_pj_D_Tj : 0);
  const gtsam::Point3 v = w_pj - w_pi;

  if (H1) {
    H1->resize(3, 6);
    H1->block<3, 6>(0, 0) = v_D_w_pi * w_pi_D_Ti;
  }

  if (H2) {
    H2->resize(3, 6);
    H2->block<3, 6>(0, 0) = v_D_w_pj * w_pj_D_Tj;
  }

  return v;
}

PointFactor::PointFactor(const gtsam::Key i, const gtsam::Key j,
                         const std::vector<PointConstraint> &constraints,
                         double sigma) noexcept
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
          gtsam::noiseModel::Diagonal::Sigmas(
              gtsam::Vector::Ones(3 * constraints.size()) * sigma),
          i, j),
      size(constraints.size()) {
  p_i.resize(3, constraints.size());
  p_j.resize(3, constraints.size());

  for (size_t index = 0; index < constraints.size(); ++index) {
    const auto &constraint = constraints[index];
    p_i.col(index) = constraint.p_i;
    p_j.col(index) = constraint.p_j;
  }
}

[[nodiscard]] gtsam::Vector
PointFactor::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                           boost::optional<gtsam::Matrix &> H1,
                           boost::optional<gtsam::Matrix &> H2) const noexcept {

  // Use broadcasted operations to compute everything at once
  Eigen::Matrix3Xd w_pi =
      (Ti.rotation().matrix() * p_i).colwise() + Ti.translation();
  Eigen::Matrix3Xd w_pj =
      (Tj.rotation().matrix() * p_j).colwise() + Tj.translation();
  Eigen::MatrixXd residual = w_pj - w_pi;
  residual.resize(3 * size, 1);

  // H1_r = R_i (p_i)_x
  // H1_t = - R_i
  if (H1) {
    H1->resize(3 * size, 6);
    Eigen::Matrix3d Ri = Ti.rotation().matrix() * -1.0f;
    for (size_t i = 0; i < size; ++i) {
      H1->block<3, 1>(3 * i, 0) = Ri.col(2) * p_i(1, i) - Ri.col(1) * p_i(2, i);
      H1->block<3, 1>(3 * i, 1) = Ri.col(0) * p_i(2, i) - Ri.col(2) * p_i(0, i);
      H1->block<3, 1>(3 * i, 2) = Ri.col(1) * p_i(0, i) - Ri.col(0) * p_i(1, i);
      H1->block<3, 3>(3 * i, 3) = Ri;
    }
  }

  // H2_r = - R_j (p_j)_x
  // H2_t = R_j
  if (H2) {
    Eigen::Matrix3d Rj = Tj.rotation().matrix();
    H2->resize(3 * size, 6);
    for (size_t i = 0; i < size; ++i) {
      H2->block<3, 1>(3 * i, 0) = Rj.col(2) * p_j(1, i) - Rj.col(1) * p_j(2, i);
      H2->block<3, 1>(3 * i, 1) = Rj.col(0) * p_j(2, i) - Rj.col(2) * p_j(0, i);
      H2->block<3, 1>(3 * i, 2) = Rj.col(1) * p_j(0, i) - Rj.col(0) * p_j(1, i);
      H2->block<3, 3>(3 * i, 3) = Rj;
    }
  }

  return residual;
}

} // namespace form::point
