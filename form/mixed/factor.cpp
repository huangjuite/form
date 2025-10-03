#include "form/mixed/factor.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>

namespace form::mixed {
// ------------------------- Separate Computation ------------------------- //
[[nodiscard]] gtsam::Vector
PlanePlane::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                          OptionalJacobian H1, OptionalJacobian H2) const noexcept {
  gtsam::Vector residual(size());

  if (H1) {
    H1->resize(size(), 6);
  }
  if (H2) {
    H2->resize(size(), 6);
  }

  // Handle the first way
  {
    Eigen::MatrixXd H1_temp, H2_temp;
    residual.segment(0, num_constraints()) =
        plane_point.evaluateError(Ti, Tj, H1 ? &H1_temp : 0, H2 ? &H2_temp : 0);
    if (H1) {
      H1->block(0, 0, plane_point.size(), 6) = H1_temp;
    }
    if (H2) {
      H2->block(0, 0, plane_point.size(), 6) = H2_temp;
    }
  }

  // Then the other way as well
  {
    gtsam::Matrix H1_temp, H2_temp;
    size_t start = num_constraints();
    residual.segment(start, size() - start) =
        point_plane.evaluateError(Tj, Ti, H2 ? &H2_temp : 0, H1 ? &H1_temp : 0);
    if (H1) {
      H1->block(start, 0, point_plane.size(), 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, point_plane.size(), 6) = H2_temp;
    }
  }

  return residual;
}
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
    H1->resize(size(), 6);
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
    H2->resize(size(), 6);
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
PointPoint::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                          OptionalJacobian H1, OptionalJacobian H2) const noexcept {
  // Use broadcasted operations to compute everything at once
  Eigen::Matrix3Xd w_pi =
      (Ti.rotation().matrix() * p_i).colwise() + Ti.translation();
  Eigen::Matrix3Xd w_pj =
      (Tj.rotation().matrix() * p_j).colwise() + Tj.translation();
  Eigen::MatrixXd residual = w_pj - w_pi;
  residual.resize(size(), 1);

  // H1_r = R_i (p_i)_x
  // H1_t = - R_i
  if (H1) {
    H1->resize(size(), 6);
    Eigen::Matrix3d Ri = Ti.rotation().matrix() * -1.0f;
    for (size_t i = 0; i < num_constraints(); ++i) {
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
    H2->resize(size(), 6);
    for (size_t i = 0; i < num_constraints(); ++i) {
      H2->block<3, 1>(3 * i, 0) = Rj.col(2) * p_j(1, i) - Rj.col(1) * p_j(2, i);
      H2->block<3, 1>(3 * i, 1) = Rj.col(0) * p_j(2, i) - Rj.col(2) * p_j(0, i);
      H2->block<3, 1>(3 * i, 2) = Rj.col(1) * p_j(0, i) - Rj.col(0) * p_j(1, i);
      H2->block<3, 3>(3 * i, 3) = Rj;
    }
  }

  return residual;
}

// ------------------------- Mixed Combined Factor ------------------------- //
size_t compute_size(const std::vector<MixedConstraint> &constraints) noexcept {
  size_t size = 0;
  for (const auto &constraint : constraints) {
    switch (constraint.type) {
    case MixedConstraint::PlanePlane:
      size += 1;
      break;
    case MixedConstraint::PlanePoint:
      size += 1;
      break;
    case MixedConstraint::PointPlane:
      size += 1;
      break;
    case MixedConstraint::PointPoint:
      size += 3;
      break;
    }
  }
  return size;
}

MixedFactor::MixedFactor(const gtsam::Key i, const gtsam::Key j,
                         const std::vector<MixedConstraint> &constraints,
                         double sigma, double weight) noexcept
    //  TODO: Figure out robust noise model later
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
          gtsam::noiseModel::Isotropic::Sigma(compute_size(constraints), sigma,
                                              false),
          i, j) {
  // Count the number of each type
  size_t plane_plane_count = 0;
  size_t plane_point_count = 0;
  size_t point_plane_count = 0;
  size_t point_point_count = 0;
  for (const auto &constraint : constraints) {
    switch (constraint.type) {
    case MixedConstraint::PlanePlane:
      plane_plane_count++;
      break;
    case MixedConstraint::PlanePoint:
      plane_point_count++;
      break;
    case MixedConstraint::PointPlane:
      point_plane_count++;
      break;
    case MixedConstraint::PointPoint:
      point_point_count++;
      break;
    }
  }

  // Fill up each type
  {
    Eigen::Matrix3Xd p_i, n_i, p_j, n_j;
    p_i.resize(3, plane_plane_count);
    n_i.resize(3, plane_plane_count);
    p_j.resize(3, plane_plane_count);
    n_j.resize(3, plane_plane_count);
    size_t idx = 0;
    for (const auto &constraint : constraints) {
      if (constraint.type == MixedConstraint::PlanePlane) {
        p_i.col(idx) = constraint.p_i;
        n_i.col(idx) = constraint.n_i;
        p_j.col(idx) = constraint.p_j;
        n_j.col(idx) = constraint.n_j;
        ++idx;
      }
    }
    plane_plane = PlanePoint(std::move(p_i), std::move(n_i), std::move(p_j));
  }

  {
    Eigen::Matrix3Xd p_i, n_i, p_j;
    p_i.resize(3, plane_point_count);
    n_i.resize(3, plane_point_count);
    p_j.resize(3, plane_point_count);
    size_t idx = 0;
    for (const auto &constraint : constraints) {
      if (constraint.type == MixedConstraint::PlanePoint) {
        p_i.col(idx) = constraint.p_i;
        n_i.col(idx) = constraint.n_i;
        p_j.col(idx) = constraint.p_j;
        ++idx;
      }
    }
    plane_point = PlanePoint(std::move(p_i), std::move(n_i), std::move(p_j));
  }

  {
    Eigen::Matrix3Xd p_i, p_j, n_j;
    p_i.resize(3, point_plane_count);
    p_j.resize(3, point_plane_count);
    n_j.resize(3, point_plane_count);
    size_t idx = 0;
    for (const auto &constraint : constraints) {
      if (constraint.type == MixedConstraint::PointPlane) {
        p_i.col(idx) = constraint.p_i;
        p_j.col(idx) = constraint.p_j;
        n_j.col(idx) = constraint.n_j;
        ++idx;
      }
    }
    point_plane = PlanePoint(std::move(p_j), std::move(n_j), std::move(p_i));
  }

  {
    Eigen::Matrix3Xd p_i, p_j;
    p_i.resize(3, point_point_count);
    p_j.resize(3, point_point_count);
    size_t idx = 0;
    for (const auto &constraint : constraints) {
      if (constraint.type == MixedConstraint::PointPoint) {
        p_i.col(idx) = constraint.p_i;
        p_j.col(idx) = constraint.p_j;
        ++idx;
      }
    }
    point_point = PointPoint(std::move(p_i), std::move(p_j));
  }

  // std::cout << "MixedFactor: Initialized with " << plane_plane.num_constraints()
  //           << " Plane-Plane, " << plane_point.num_constraints() << " Plane-Point,
  //           "
  //           << point_plane.num_constraints() << " Point-Plane, "
  //           << point_point.num_constraints() << " Point-Point constraints."
  //           << std::endl;
}

[[nodiscard]] gtsam::Vector
MixedFactor::evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                           boost::optional<gtsam::Matrix &> H1,
                           boost::optional<gtsam::Matrix &> H2) const noexcept {

  size_t size = plane_plane.size() + plane_point.size() + point_plane.size() +
                point_point.size();
  gtsam::Vector residual(size);

  if (H1) {
    H1->resize(size, 6);
  }
  if (H2) {
    H2->resize(size, 6);
  }

  // Plane-Plane constraints
  size_t start = 0, end = plane_plane.size();
  if (plane_plane.size() > 0) {
    Eigen::MatrixXd H1_temp, H2_temp;
    residual.segment(start, end - start) =
        plane_plane.evaluateError(Ti, Tj, H1 ? &H1_temp : 0, H2 ? &H2_temp : 0);
    if (H1) {
      H1->block(start, 0, end - start, 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, end - start, 6) = H2_temp;
    }
  }
  start = end;

  // Plane-Point constraints
  end += plane_point.size();
  if (plane_point.size() > 0) {
    Eigen::MatrixXd H1_temp, H2_temp;
    residual.segment(start, end - start) =
        plane_point.evaluateError(Ti, Tj, H1 ? &H1_temp : 0, H2 ? &H2_temp : 0);
    if (H1) {
      H1->block(start, 0, end - start, 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, end - start, 6) = H2_temp;
    }
  }
  start = end;

  // Point-Plane constraints
  end += point_plane.size();
  if (point_plane.size() > 0) {
    Eigen::MatrixXd H1_temp, H2_temp;
    // Input is backwards here
    residual.segment(start, end - start) =
        point_plane.evaluateError(Tj, Ti, H2 ? &H2_temp : 0, H1 ? &H1_temp : 0);
    if (H1) {
      H1->block(start, 0, end - start, 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, end - start, 6) = H2_temp;
    }
  }
  start = end;

  // Point-Point constraints
  end += point_point.size();
  if (point_point.size() > 0) {
    Eigen::MatrixXd H1_temp, H2_temp;
    residual.segment(start, end - start) =
        point_point.evaluateError(Ti, Tj, H1 ? &H1_temp : 0, H2 ? &H2_temp : 0);
    if (H1) {
      H1->block(start, 0, end - start, 6) = H1_temp;
    }
    if (H2) {
      H2->block(start, 0, end - start, 6) = H2_temp;
    }
  }

  return residual;
}

} // namespace form::mixed
