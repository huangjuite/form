#include "form/classic/factor.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

namespace form::classic {
// ------------------------- Classic Computation ------------------------- //

[[nodiscard]] gtsam::Vector
PlanePoint::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                          OptionalJacobian H1, OptionalJacobian H2) const noexcept {
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
    H1->resize(num_residuals(), 6);
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
    H2->resize(num_residuals(), 6);
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

[[nodiscard]] gtsam::Vector
EdgePoint::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                         OptionalJacobian H1, OptionalJacobian H2) const noexcept {
  // Use broadcasted operations to compute everything at once
  Eigen::Matrix3Xd w_pi =
      (Ti.rotation().matrix() * p_i).colwise() + Ti.translation();
  Eigen::Matrix3Xd w_di = Ti.rotation().matrix() * d_i;
  Eigen::Matrix3Xd w_pj =
      (Tj.rotation().matrix() * p_j).colwise() + Tj.translation();
  Eigen::Matrix3Xd delta = w_pj - w_pi;

  // Eigen::MatrixXd residual = (w_pj - w_pi).cross(w_di);
  // residual.resize(num_residuals(), 1);
  Eigen::VectorXd residual(num_residuals());
  for (size_t i = 0; i < num_constraints(); ++i) {
    residual.segment(3 * i, 3) = delta.col(i).cross(w_di.col(i));
  }

  // H1_r = - R_i (p_i - d_i)_x
  // H1_t = R_i
  if (H1) {
    Eigen::Matrix3d Ri = Ti.rotation().matrix();
    H1->resize(num_residuals(), 6);

    for (size_t i = 0; i < num_constraints(); ++i) {
      Eigen::Matrix3d cross = gtsam::skewSymmetric(Ri * d_i.col(i));
      H1->block(3 * i, 0, 3, 3) =
          -cross * Ri * gtsam::skewSymmetric(p_i.col(i)) -
          gtsam::skewSymmetric(delta.col(i)) * Ri * gtsam::skewSymmetric(d_i.col(i));
      H1->block(3 * i, 3, 3, 3) = cross * Ri;
    }
  }

  // H2_r = - R_j (p_j)_x
  // H2_t = R_j
  if (H2) {
    Eigen::Matrix3d Ri = Ti.rotation().matrix();
    Eigen::Matrix3d Rj = Tj.rotation().matrix();
    H2->resize(num_residuals(), 6);

    // TODO: Can we speed this up at all?
    for (size_t i = 0; i < num_constraints(); ++i) {
      Eigen::Matrix3d cross = gtsam::skewSymmetric(Ri * d_i.col(i));
      H2->block(3 * i, 0, 3, 3) = cross * Rj * gtsam::skewSymmetric(p_j.col(i));
      H2->block(3 * i, 3, 3, 3) = -cross * Rj;
    }
  }

  return residual;
}

// ------------------------- Classic Combined Factor ------------------------- //
gtsam::Vector make_sigma(const std::vector<ClassicConstraint> &constraints,
                         double sigma) noexcept {
  size_t size_planar = 0;
  size_t size_point = 0;
  for (const auto &constraint : constraints) {
    switch (constraint.kind) {
    case ClassicConstraint::Plane:
      size_planar += 1;
      break;
    case ClassicConstraint::Edge:
      size_point += 3;
      break;
    }
  }

  gtsam::Vector sigma_vector(size_planar + size_point);
  // TODO: Equally weighted for now
  sigma_vector.head(size_planar).setConstant(sigma);
  sigma_vector.tail(size_point).setConstant(sigma);
  return sigma_vector;
}

ClassicFactor::ClassicFactor(const gtsam::Key i, const gtsam::Key j,
                             const std::vector<ClassicConstraint> &constraints,
                             double sigma) noexcept
    //  TODO: Figure out robust noise model later
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
          // gtsam::noiseModel::Robust::Create(
          //     gtsam::noiseModel::mEstimator::Huber::Create(
          //         1.345, gtsam::noiseModel::mEstimator::Base::Scalar),
          gtsam::noiseModel::Diagonal::Sigmas(make_sigma(constraints, sigma)), i,
          j) {
  // Count the number of each type
  size_t plane_count = 0;
  size_t edge_count = 0;
  for (const auto &constraint : constraints) {
    switch (constraint.kind) {
    case ClassicConstraint::Plane:
      plane_count++;
      break;
    case ClassicConstraint::Edge:
      edge_count++;
      break;
    }
  }

  // Fill up each type
  {
    Eigen::Matrix3Xd p_i, n_i, p_j;
    p_i.resize(3, plane_count);
    n_i.resize(3, plane_count);
    p_j.resize(3, plane_count);
    size_t idx = 0;
    for (const auto &constraint : constraints) {
      if (constraint.kind == ClassicConstraint::Plane) {
        p_i.col(idx) = constraint.p_i;
        n_i.col(idx) = constraint.n_i;
        p_j.col(idx) = constraint.p_j;
        ++idx;
      }
    }
    plane = PlanePoint(std::move(p_i), std::move(n_i), std::move(p_j));
  }

  {
    Eigen::Matrix3Xd p_i, d_i, p_j;
    p_i.resize(3, edge_count);
    d_i.resize(3, edge_count);
    p_j.resize(3, edge_count);
    size_t idx = 0;
    for (const auto &constraint : constraints) {
      if (constraint.kind == ClassicConstraint::Edge) {
        p_i.col(idx) = constraint.p_i;
        d_i.col(idx) = constraint.n_i;
        p_j.col(idx) = constraint.p_j;
        ++idx;
      }
    }
    edge = EdgePoint(std::move(p_i), std::move(d_i), std::move(p_j));
  }
}

[[nodiscard]] gtsam::Vector
ClassicFactor::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                             boost::optional<gtsam::Matrix &> H1,
                             boost::optional<gtsam::Matrix &> H2) const noexcept {

  size_t size = plane.num_residuals() + edge.num_residuals();
  gtsam::Vector residual(size);

  if (H1) {
    H1->resize(size, 6);
  }
  if (H2) {
    H2->resize(size, 6);
  }

  // Plane-Plane constraints
  size_t start = 0, end = plane.num_residuals();
  if (plane.num_residuals() > 0) {
    Eigen::MatrixXd H1_temp, H2_temp;
    residual.segment(start, end - start) =
        plane.evaluateError(Ti, Tj, H1 ? &H1_temp : 0, H2 ? &H2_temp : 0);
    if (H1) {
      H1->block(start, 0, end - start, 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, end - start, 6) = H2_temp;
    }
  }
  start = end;

  // Edge constraints
  end += edge.num_residuals();
  if (edge.num_residuals() > 0) {
    Eigen::MatrixXd H1_temp, H2_temp;
    residual.segment(start, end - start) =
        edge.evaluateError(Ti, Tj, H1 ? &H1_temp : 0, H2 ? &H2_temp : 0);
    if (H1) {
      H1->block(start, 0, end - start, 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, end - start, 6) = H2_temp;
    }
  }

  return residual;
}

} // namespace form::classic
