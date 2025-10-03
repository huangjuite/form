#pragma once

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>

namespace form::point {

// Forward declarations
struct PointFactor;
struct PointConstraint;
struct ExtractPoint;

// For storing in local map
template <typename T, typename S = size_t> struct PointXYZS {
  using type_t = T;

  typedef PointXYZS<T, S> Self;
  typedef PointFactor Factor;
  typedef PointConstraint Constraint;
  typedef ExtractPoint Extractor;

  T x;
  T y;
  T z;
  T _ = static_cast<T>(0);
  S scan;

  // ------------------------- Constructor ------------------------- //
  PointXYZS() = default;

  PointXYZS(T x, T y, T z, S scan) : x(x), y(y), z(z), _(0), scan(scan) {}

  // ------------------------- Position getters ------------------------- //
  [[nodiscard]] inline Eigen::Map<const Eigen::Matrix<T, 3, 1>>
  vec3() const noexcept {
    return Eigen::Map<const Eigen::Matrix<T, 3, 1>>(&x);
  }

  [[nodiscard]] inline Eigen::Map<Eigen::Matrix<T, 3, 1>> vec3() noexcept {
    return Eigen::Map<Eigen::Matrix<T, 3, 1>>(&x);
  }

  [[nodiscard]] inline Eigen::Map<const Eigen::Matrix<T, 4, 1>>
  vec4() const noexcept {
    return Eigen::Map<const Eigen::Matrix<T, 4, 1>>(&x);
  }

  [[nodiscard]] inline Eigen::Map<const Eigen::Array<T, 3, 1>>
  array() const noexcept {
    return Eigen::Map<const Eigen::Array<T, 3, 1>>(&x);
  }

  [[nodiscard]] inline Eigen::Map<Eigen::Array<T, 3, 1>> array() noexcept {
    return Eigen::Map<Eigen::Array<T, 3, 1>>(&x);
  }

  // ------------------------- Misc ------------------------- //
  inline void transform_in_place(const gtsam::Pose3 &pose) noexcept {
    vec3() = (pose * vec3().template cast<double>()).template cast<T>();
  }

  [[nodiscard]] inline PointXYZS<T, S>
  transform(const gtsam::Pose3 &pose) const noexcept {
    auto point = *this;
    point.transform_in_place(pose);
    return point;
  }

  [[nodiscard]] inline T squaredNorm() const noexcept {
    return vec4().squaredNorm();
  }

  [[nodiscard]] inline T norm() const noexcept { return vec4().norm(); }

  [[nodiscard]] constexpr bool operator==(const PointXYZS &other) const noexcept {
    return x == other.x && y == other.y && z == other.z && scan == other.scan;
  }
};

} // namespace form::point