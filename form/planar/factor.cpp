#include "form/planar/factor.hpp"
#include <gtsam/base/Matrix.h>

namespace form::planar {

// TODO: Probably easier to just split this into two functions
[[nodiscard]] gtsam::Vector
PlanarConstraint::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                                bool use_plane_to_plane,
                                boost::optional<gtsam::Matrix &> H1,
                                boost::optional<gtsam::Matrix &> H2) const noexcept {
  gtsam::Matrix33 w_ni_D_Ri;
  gtsam::Matrix36 w_pi_D_Ti;
  gtsam::Matrix33 w_nj_D_Rj;
  gtsam::Matrix36 w_pj_D_Tj;
  gtsam::Matrix13 r1_D_w_ni;
  gtsam::Matrix13 r1_D_v;
  gtsam::Matrix13 r2_D_w_nj;
  gtsam::Matrix13 r2_D_v;
  const double v_D_w_pj = 1.0;
  const double v_D_w_pi = -1.0;

  const gtsam::Point3 w_ni = Ti.rotation().rotate(n_i, H1 ? &w_ni_D_Ri : 0);
  const gtsam::Point3 w_pi = Ti.transformFrom(p_i, H1 ? &w_pi_D_Ti : 0);
  const gtsam::Point3 w_nj = Tj.rotation().rotate(n_j, H2 ? &w_nj_D_Rj : 0);
  const gtsam::Point3 w_pj = Tj.transformFrom(p_j, H2 ? &w_pj_D_Tj : 0);
  const gtsam::Point3 v = w_pj - w_pi;

  double r1 = gtsam::dot(w_ni, v, H1 ? &r1_D_w_ni : 0, H1 || H2 ? &r1_D_v : 0);

  if (H1) {
    H1->resize(use_plane_to_plane ? 2 : 1, 6);
    H1->block<1, 6>(0, 0) = r1_D_v * v_D_w_pi * w_pi_D_Ti;
    H1->block<1, 3>(0, 0) += r1_D_w_ni * w_ni_D_Ri;
  }

  if (H2) {
    H2->resize(use_plane_to_plane ? 2 : 1, 6);
    H2->block<1, 6>(0, 0) = r1_D_v * v_D_w_pj * w_pj_D_Tj;
  }

  return (gtsam::Vector(1) << r1).finished();
}

PlanarFactor::PlanarFactor(const gtsam::Key i, const gtsam::Key j,
                           const std::vector<PlanarConstraint> &constraints,
                           double sigma) noexcept
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(
                  1.345, gtsam::noiseModel::mEstimator::Base::Scalar),
              gtsam::noiseModel::Diagonal::Sigmas(
                  gtsam::Vector::Ones(constraints.size()) * sigma)),
          i, j),
      m_use_plane_to_plane(false), size(constraints.size()) {
  p_i.resize(3, constraints.size());
  p_j.resize(3, constraints.size());
  n_i.resize(3, constraints.size());
  n_j.resize(3, constraints.size());

  for (size_t index = 0; index < constraints.size(); ++index) {
    const auto &constraint = constraints[index];
    p_i.col(index) = constraint.p_i;
    p_j.col(index) = constraint.p_j;
    n_i.col(index) = constraint.n_i;
    n_j.col(index) = constraint.n_j;
  }
}

[[nodiscard]] gtsam::Vector
PlanarFactor::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                            boost::optional<gtsam::Matrix &> H1,
                            boost::optional<gtsam::Matrix &> H2) const noexcept {

  // Use broadcasted operations to compute everything at once
  Eigen::Matrix3Xd w_ni = Ti.rotation().matrix() * n_i;
  Eigen::Matrix3Xd w_pi =
      (Ti.rotation().matrix() * p_i).colwise() + Ti.translation();
  Eigen::Matrix3Xd w_pj =
      (Tj.rotation().matrix() * p_j).colwise() + Tj.translation();
  Eigen::Matrix3Xd v = w_pj - w_pi;
  gtsam::Vector residual = (w_ni.array() * v.array()).colwise().sum();

  // H1_r = w_ni^T R_i (p_i)_x + v^T R_i (n_i)_x
  // H1_t = - w_ni^T R_i
  if (H1) {
    H1->resize(size, 6);
    Eigen::Matrix3Xd RT_n = Ti.rotation().transpose() * w_ni; // 3xn
    Eigen::Matrix3Xd RT_v = Ti.rotation().transpose() * v;    // 3xn
    H1->col(0) =
        RT_n.row(1).cwiseProduct(p_i.row(2)) - RT_n.row(2).cwiseProduct(p_i.row(1)) -
        RT_v.row(1).cwiseProduct(n_i.row(2)) + RT_v.row(2).cwiseProduct(n_i.row(1));
    H1->col(1) =
        RT_n.row(2).cwiseProduct(p_i.row(0)) - RT_n.row(0).cwiseProduct(p_i.row(2)) -
        RT_v.row(2).cwiseProduct(n_i.row(0)) + RT_v.row(0).cwiseProduct(n_i.row(2));
    H1->col(2) =
        RT_n.row(0).cwiseProduct(p_i.row(1)) - RT_n.row(1).cwiseProduct(p_i.row(0)) -
        RT_v.row(0).cwiseProduct(n_i.row(1)) + RT_v.row(1).cwiseProduct(n_i.row(0));
    H1->rightCols(3) = -RT_n.transpose();
  }

  // H2_r = - w_ni^T R_j (p_j)_x
  // H2_t = w_ni^T R_j
  if (H2) {
    H2->resize(size, 6);
    Eigen::Matrix3Xd RT_n = Tj.rotation().transpose() * w_ni; // 3xn
    H2->col(0) =
        -RT_n.row(1).cwiseProduct(p_j.row(2)) + RT_n.row(2).cwiseProduct(p_j.row(1));
    H2->col(1) =
        -RT_n.row(2).cwiseProduct(p_j.row(0)) + RT_n.row(0).cwiseProduct(p_j.row(2));
    H2->col(2) =
        -RT_n.row(0).cwiseProduct(p_j.row(1)) + RT_n.row(1).cwiseProduct(p_j.row(0));
    H2->rightCols(3) = RT_n.transpose();
  }

  return residual;
}

PlanarFactor1::PlanarFactor1(const gtsam::Pose3 Ti, const gtsam::Key j,
                             const std::vector<PlanarConstraint> &constraints,
                             double sigma, bool use_plane_to_plane) noexcept
    : gtsam::NoiseModelFactor1<gtsam::Pose3>(
          // gtsam::noiseModel::Robust::Create(
          //     gtsam::noiseModel::mEstimator::Huber::Create(
          //         1.345, gtsam::noiseModel::mEstimator::Base::Scalar),
          gtsam::noiseModel::Diagonal::Sigmas(
              gtsam::Vector::Ones(use_plane_to_plane ? 2 * constraints.size()
                                                     : constraints.size()) *
              sigma),
          j),
      m_constraints(constraints), m_use_plane_to_plane(use_plane_to_plane),
      m_Ti(Ti) {}

[[nodiscard]] gtsam::Vector
PlanarFactor1::evaluateError(const gtsam::Pose3 &Tj,
                             boost::optional<gtsam::Matrix &> H2) const noexcept {

  long size = m_use_plane_to_plane ? 2 * m_constraints.size() : m_constraints.size();
  gtsam::Vector residual(size);

  gtsam::Matrix H1_val;
  gtsam::Matrix H2_val;
  boost::optional<gtsam::Matrix &> H1_opt = boost::none;
  boost::optional<gtsam::Matrix &> H2_opt = boost::none;

  if (H2) {
    H2->resize(size, 6);
    H2_opt = H2_val;
  }

  for (long index = 0; index < m_constraints.size(); ++index) {
    gtsam::Vector r_i = m_constraints[index].evaluateError(
        m_Ti, Tj, m_use_plane_to_plane, H1_opt, H2_opt);

    if (m_use_plane_to_plane) {
      residual.block<2, 1>(2 * index, 0) = r_i;
      if (H2)
        H2->block<2, 6>(2 * index, 0) = *H2_opt;
    } else {
      residual(index) = r_i(0);
      if (H2)
        H2->block<1, 6>(index, 0) = *H2_opt;
    }
  }

  return residual;
}

} // namespace form::planar
