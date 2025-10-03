#pragma once

#include "form/point/feature.hpp"
#include "form/point_types.hpp"
#include "tsl/robin_map.h"
#include <Eigen/Dense>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace form::point {

inline Eigen::Matrix<int, 3, 1>
computeCoords(const PointXYZS<double> &point,
              typename PointXYZS<double>::type_t voxel_width) noexcept {
  return (point.array() / voxel_width).floor().template cast<int>();
}

inline std::vector<PointXYZS<double>>
VoxelDownsample(const std::vector<PointXYZS<double>> &frame,
                const double voxel_size) {
  tsl::robin_map<Eigen::Matrix<int, 3, 1>, PointXYZS<double>> grid;

  std::for_each(frame.cbegin(), frame.cend(), [&](const auto &point) {
    const auto voxel = computeCoords(point, voxel_size);
    if (!grid.contains(voxel))
      grid.insert({voxel, point});
  });
  std::vector<PointXYZS<double>> frame_dowsampled;
  frame_dowsampled.reserve(grid.size());
  std::for_each(grid.cbegin(), grid.cend(), [&](const auto &voxel_and_point) {
    frame_dowsampled.emplace_back(voxel_and_point.second);
  });
  return frame_dowsampled;
}

template <typename Point>
std::vector<PointXYZS<double>>
extract_keypoints_point(const PointCloud<Point> &frame, size_t scan_idx) {
  std::vector<PointXYZS<double>> points;
  points.reserve(frame.points.size());
  std::for_each(frame.points.cbegin(), frame.points.cend(), [&](const auto &point) {
    points.push_back(PointXYZS<double>(point.x, point.y, point.z, scan_idx));
  });

  const auto voxel_size = 1.0;
  const std::vector<PointXYZS<double>> frame_downsample =
      VoxelDownsample(points, voxel_size * 0.5);
  const std::vector<PointXYZS<double>> source =
      VoxelDownsample(frame_downsample, voxel_size * 1.5);

  return source;
}

struct ExtractPoint {
  struct Params {
    double voxel_width = 1.0;

    double min_norm_squared = 1.0;
    double max_norm_squared = 100.0 * 100.0;
  };
  Params params;

  ExtractPoint(const Params &params) : params(params) {}

  template <typename Point>
  std::vector<PointXYZS<double>> operator()(const PointCloud<Point> &scan,
                                            size_t scan_idx) const {
    return extract_keypoints_point(scan, scan_idx);
  }
};

} // namespace form::point
