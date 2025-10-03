#pragma once

#include "form/classic/feature.hpp"
#include "form/point_types.hpp"
#include <Eigen/Dense>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>

namespace form::classic {

struct KeypointExtractionParams {
  // Parameters for keypoint extraction
  size_t neighbor_points = 5;
  size_t num_sectors = 6;
  size_t planar_feats_per_sector = 50;
  double planar_threshold = 1.0;
  size_t edge_feats_per_sector = 10;
  double edge_feat_threshold = 100.0;
  // TODO: One of these isn't used - remove it
  double occlusion_thresh = 0.9;

  // Parameters for normal estimation
  double radius = 1.0;
  size_t min_points = 5;

  // Based on LiDAR info
  double min_norm_squared = 1.0;
  double max_norm_squared = 100.0 * 100.0;
};

template <typename Point>
std::vector<bool> compute_in_range_points(const PointCloud<Point> &scan,
                                          const KeypointExtractionParams &params) {
  using T = typename Point::type_t;
  using channel_t = decltype(Point::channel);

  std::vector<bool> mask(scan.points.size(), true);
  size_t num_points = scan.points.size();

  for (size_t scan_line_idx = 0; scan_line_idx < scan.num_rows; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < scan.num_columns; line_pt_idx++) {
      const size_t idx = (scan_line_idx * scan.num_columns) + line_pt_idx;

      // CHECK 2: Is the point in the valid range of the LiDAR
      const Point &point = scan.points[idx];
      const double range2 = point.squaredNorm();
      if (range2 < params.min_norm_squared || range2 > params.max_norm_squared) {
        mask[idx] = false;
        continue;
      }
    }
  }

  return mask;
}

template <typename Point>
std::vector<bool> compute_valid_points(const PointCloud<Point> &scan,
                                       const KeypointExtractionParams &params) {
  using T = typename Point::type_t;
  using channel_t = decltype(Point::channel);

  if (scan.points.size() != scan.num_columns * scan.num_rows) {
    throw std::runtime_error("Provided scan does not match the expected size " +
                             std::to_string(scan.num_columns * scan.num_rows) +
                             " != " + std::to_string(scan.points.size()));
  }

  std::vector<bool> mask(scan.points.size(), true);
  size_t num_points = scan.points.size();

  // Compute the valid points based on the parameters
  // Structured search (search over each scan line individually over all points
  // [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < scan.num_rows; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < scan.num_columns; line_pt_idx++) {
      const size_t idx = (scan_line_idx * scan.num_columns) + line_pt_idx;

      // CHECK 1: Due to edge effects, the first and last neighbor_points points
      // of each scan line are invalid
      if (line_pt_idx < params.neighbor_points ||
          line_pt_idx >= scan.num_columns - params.neighbor_points) {
        mask[idx] = false;
        continue;
      }

      // CHECK 2: Is the point in the valid range of the LiDAR
      const Point &point = scan.points[idx];
      const double range2 = point.squaredNorm();
      if (range2 < params.min_norm_squared || range2 > params.max_norm_squared) {
        mask[idx] = false;
        for (size_t i = 1; i <= params.neighbor_points; i++) {
          mask[idx - i] = false;
          mask[idx + i] = false;
        }
        continue;
      }

      // Get the current point and its two neighbors
      const Point &prev_point = scan.points[idx - 1];
      const Point &next_point = scan.points[idx + 1];

      // Compute the range of each point
      const double range = std::sqrt(range2);
      const double next_range = next_point.norm();
      const double prev_range = prev_point.norm();

      // TODO: I'm not sure how necessary checks 3/4 are when only detecting
      // planar features CHECK 3: Occlusions
      if (next_range - range > params.occlusion_thresh) { // Case 1
        for (size_t n = 1; n <= params.neighbor_points; n++) {
          mask[idx + n] = false;
        }
        continue;
      } else if (range - next_range > params.occlusion_thresh) { // Case 2
        for (size_t n = 0; n < params.neighbor_points; n++) {
          mask[idx - n] = false;
        }
        continue;
      }

      // CHECK 4: Check if the point is on a plane nearly parallel to the LiDAR
      // Beam (no continue b/c last )
      // double diff_next = std::abs(prev_range - range);
      // double diff_prev = std::abs(next_range - range);
      // if (diff_next > params.parallel_thresh * range &&
      //     diff_prev > params.parallel_thresh * range) {
      //   mask[idx] = false;
      // }
    } // end line point search
  } // end scan line search

  return mask;
}

/// @brief Structure for storing curvature information for points
template <typename T> struct Curvature {
  /// @brief The index of the point
  size_t index;
  /// @brief The curvature of the point
  T curvature;
  /// @brief Explicit parameterized constructor
  Curvature(size_t index, double curvature) : index(index), curvature(curvature) {}
  /// @brief Default constructor
  Curvature() = default;

  /// @brief Comparison operator for sorting
  bool operator<(const Curvature &other) const {
    return curvature < other.curvature;
  }
};

template <typename Point>
std::vector<Curvature<typename Point::type_t>>
compute_curvature(const PointCloud<Point> &scan, const std::vector<bool> &mask,
                  const KeypointExtractionParams &params, size_t scan_idx) noexcept {

  using T = typename Point::type_t;
  std::vector<Curvature<T>> curvature;

  // Structured search (search over each scan line individually over all points
  // [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < scan.num_rows; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < scan.num_columns; line_pt_idx++) {
      const size_t idx = (scan_line_idx * scan.num_columns) + line_pt_idx;
      // If not valid, input max curvature
      if (!mask[idx]) {
        curvature.emplace_back(idx, std::numeric_limits<T>::max());
      }
      // If valid compute the curvature
      else {
        // Initialize with the difference term
        double r2 = scan.points[idx].squaredNorm();
        double dx = -(2.0 * params.neighbor_points) * scan.points[idx].x;
        double dy = -(2.0 * params.neighbor_points) * scan.points[idx].y;
        double dz = -(2.0 * params.neighbor_points) * scan.points[idx].z;
        // Iterate over neighbors and accumulate
        for (size_t n = 1; n <= params.neighbor_points; n++) {
          dx = dx + scan.points[idx - n].x + scan.points[idx + n].x;
          dy = dy + scan.points[idx - n].y + scan.points[idx + n].y;
          dz = dz + scan.points[idx - n].z + scan.points[idx + n].z;
        }
        curvature.emplace_back(idx, dx * dx + dy * dy + dz * dz);
        // curvature.emplace_back(idx, (dx * dx + dy * dy + dz * dz) / r2);
      }
    }
  }
  return curvature;
}

template <typename T>
void extract_planar(const size_t &sector_start_point, const size_t &sector_end_point,
                    const std::vector<Curvature<T>> &curvature,
                    const KeypointExtractionParams &params,
                    std::vector<size_t> &out_features,
                    std::vector<bool> &valid_mask) {

  size_t num_sector_planar_features = 0;
  // Iterate through all points in the sector
  for (size_t sorted_curv_idx = sector_start_point;
       sorted_curv_idx < sector_end_point; sorted_curv_idx++) {
    const Curvature curv = curvature[sorted_curv_idx];
    if (valid_mask[curv.index] && curv.curvature < params.planar_threshold) {
      out_features.push_back(curv.index);
      // mark the neighbors as used so they aren't also added in
      for (size_t n = 0; n < params.neighbor_points; n++) {
        valid_mask[curv.index + n] = false;
        valid_mask[curv.index - n] = false;
      }
      num_sector_planar_features++;
    }
    // Early exit if we have found enough features
    if (num_sector_planar_features > params.planar_feats_per_sector)
      break;

  } // end feature search in sector
}

template <typename T>
void extract_edge(const size_t &sector_start_point, const size_t &sector_end_point,
                  const std::vector<Curvature<T>> &curvature,
                  const KeypointExtractionParams &params,
                  std::vector<size_t> &out_features, std::vector<bool> &valid_mask) {
  size_t num_sector_edge_features = 0;
  for (size_t sorted_curv_idx_p1 = sector_end_point;
       sorted_curv_idx_p1 > sector_start_point; sorted_curv_idx_p1--) {
    const Curvature<T> curv =
        curvature[sorted_curv_idx_p1 - 1]; // subtraction as loop cannot go negative

    if (valid_mask[curv.index] && curv.curvature > params.edge_feat_threshold) {
      out_features.push_back(curv.index);
      for (size_t n = 0; n < params.neighbor_points; n++) { // update mask
        valid_mask[curv.index + n] = false;
        valid_mask[curv.index - n] = false;
      }
      num_sector_edge_features++;
    }
    // Early exit if we have found enough features
    if (num_sector_edge_features > params.edge_feats_per_sector)
      break;
  }
}

template <typename Point>
std::optional<size_t> find_closest(const Point &point, const size_t &start,
                                   const size_t &end, const PointCloud<Point> &scan,
                                   const std::vector<bool> &valid_mask) {
  std::optional<size_t> closest_point = std::nullopt;
  double min_dist2 = std::numeric_limits<double>::max();
  for (size_t idx = start; idx < end; idx++) {
    if (!valid_mask[idx]) {
      continue;
    }
    const double dist2 = (scan.points[idx].vec4() - point.vec4()).squaredNorm();
    if (dist2 < min_dist2) {
      min_dist2 = dist2;
      closest_point = idx;
    }
  }
  return closest_point;
}

template <typename Point>
void find_neighbors(const size_t &idx, const PointCloud<Point> &scan,
                    const KeypointExtractionParams &params,
                    std::vector<Point> &out) {
  // search in the positive direction
  const auto &point = scan.points[idx];
  for (size_t i = 1; i <= params.neighbor_points; i++) {
    const auto &neighbor = scan.points[idx + i];
    const double range2 = (neighbor.vec4() - point.vec4()).squaredNorm();
    if (range2 < params.radius * params.radius) {
      out.push_back(neighbor);
    } else {
      break;
    }
  }

  // search in the negative direction
  for (size_t i = 1; i <= params.neighbor_points; i++) {
    const auto &neighbor = scan.points[idx - i];
    const double range2 = (neighbor.vec4() - point.vec4()).squaredNorm();
    if (range2 < params.radius * params.radius) {
      out.push_back(neighbor);
    } else {
      break;
    }
  }
}

template <typename Point>
std::optional<Eigen::Matrix<typename Point::type_t, 3, 1>>
compute_normal(const size_t &idx, const PointCloud<Point> &scan,
               const KeypointExtractionParams &params,
               const std::vector<bool> &valid_mask,
               const size_t cov_idx = 0) noexcept {
  using T = typename Point::type_t;
  const size_t scan_line_idx = idx / scan.num_columns;
  const auto start = scan.points.cbegin();
  const auto end = scan.points.cend();
  const auto &point = scan.points[idx];

  // First find neighbors on own scan line
  std::vector<Point> neighbors;
  find_neighbors(idx, scan, params, neighbors);

  bool found_other_scanline = false;

  // Get the neighbors of the point on the previous scan line
  if (scan_line_idx > 0) {
    const size_t prev_scan_line_idx = scan_line_idx - 1;
    const auto closest_idx =
        find_closest(point, scan.num_columns * prev_scan_line_idx,
                     scan.num_columns * (prev_scan_line_idx + 1), scan, valid_mask);
    if (closest_idx.has_value()) {
      found_other_scanline = true;
      neighbors.push_back(scan.points[*closest_idx]);
      find_neighbors(*closest_idx, scan, params, neighbors);
    }
  }

  // Get the neighbors of the point on the next scan line
  if (scan_line_idx < scan.num_rows - 1) {
    // std::printf("---- Searching next scan line %zu\n", scan_line_idx + 1);
    const size_t next_scan_line_idx = scan_line_idx + 1;
    const auto closest_idx =
        find_closest(point, scan.num_columns * next_scan_line_idx,
                     scan.num_columns * (next_scan_line_idx + 1), scan, valid_mask);
    if (closest_idx.has_value()) {
      found_other_scanline = true;
      neighbors.push_back(scan.points[*closest_idx]);
      find_neighbors(closest_idx.value(), scan, params, neighbors);
    }
  }

  // If there's not enough neighbors, return failed
  if (!found_other_scanline || neighbors.size() < params.min_points) {
    return std::nullopt;
  }

  // Compute the covariance matrix
  // std::printf("---- Found %zu neighbors\n", neighbors.size());
  Eigen::Matrix<T, Eigen::Dynamic, 3> A(neighbors.size(), 3);
  for (size_t j = 0; j < neighbors.size(); ++j) {
    A.row(j) = neighbors[j].vec3() - point.vec3();
  }
  A /= neighbors.size();
  Eigen::Matrix<T, 3, 3> Cov = A.transpose() * A;

  // Eigenvalues + normals
  // std::printf("---- computing eigenvalues\n");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> b(
      Cov, Eigen::ComputeEigenvectors);
  Eigen::Matrix<T, 3, 1> normal = b.eigenvectors().col(cov_idx);
  normal.normalize();

  return normal;
}

template <typename Point>
void find_edge_neighbors(const size_t &idx, const PointCloud<Point> &scan,
                         const KeypointExtractionParams &params,
                         const std::vector<bool> &valid_mask,
                         std::vector<Point> &out) {
  for (size_t i = idx - params.neighbor_points; i < idx + params.neighbor_points;
       i++) {
    if (i < 0 || i >= scan.points.size()) {
      continue; // Skip out of bounds indices
    }
    if (!valid_mask[i]) {
      continue; // Skip invalid points
    }
    out.push_back(scan.points[i]);
  }
}

template <typename Point>
std::optional<Eigen::Matrix<typename Point::type_t, 3, 1>>
compute_edge_direction(const size_t &idx, const PointCloud<Point> &scan,
                       const KeypointExtractionParams &params,
                       const std::vector<bool> &valid_mask) noexcept {
  using T = typename Point::type_t;
  const size_t scan_line_idx = idx / scan.num_columns;
  const auto start = scan.points.cbegin();
  const auto end = scan.points.cend();
  const auto &point = scan.points[idx];

  // First find neighbors on own scan line
  std::vector<Point> neighbors;
  find_edge_neighbors(idx, scan, params, valid_mask, neighbors);

  bool found_other_scanline = false;

  // Get the neighbors of the point on the previous scan line
  if (scan_line_idx > 0) {
    const size_t prev_scan_line_idx = scan_line_idx - 1;
    const auto closest_idx =
        find_closest(point, scan.num_columns * prev_scan_line_idx,
                     scan.num_columns * (prev_scan_line_idx + 1), scan, valid_mask);
    if (closest_idx.has_value()) {
      found_other_scanline = true;
      neighbors.push_back(scan.points[*closest_idx]);
      find_edge_neighbors(*closest_idx, scan, params, valid_mask, neighbors);
    }
  }

  // Get the neighbors of the point on the next scan line
  if (scan_line_idx < scan.num_rows - 1) {
    // std::printf("---- Searching next scan line %zu\n", scan_line_idx + 1);
    const size_t next_scan_line_idx = scan_line_idx + 1;
    const auto closest_idx =
        find_closest(point, scan.num_columns * next_scan_line_idx,
                     scan.num_columns * (next_scan_line_idx + 1), scan, valid_mask);
    if (closest_idx.has_value()) {
      found_other_scanline = true;
      neighbors.push_back(scan.points[*closest_idx]);
      find_edge_neighbors(closest_idx.value(), scan, params, valid_mask, neighbors);
    }
  }

  // If there's not enough neighbors, return failed
  if (!found_other_scanline || neighbors.size() < params.min_points) {
    return std::nullopt;
  }

  // Compute the covariance matrix
  // std::printf("---- Found %zu neighbors\n", neighbors.size());
  Eigen::Matrix<T, Eigen::Dynamic, 3> A(neighbors.size(), 3);
  for (size_t j = 0; j < neighbors.size(); ++j) {
    A.row(j) = neighbors[j].vec3() - point.vec3();
  }
  A /= neighbors.size();
  Eigen::Matrix<T, 3, 3> Cov = A.transpose() * A;

  // Eigenvalues + normals
  // std::printf("---- computing eigenvalues\n");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> b(
      Cov, Eigen::ComputeEigenvectors);
  Eigen::Matrix<T, 3, 1> normal = b.eigenvectors().col(0);
  normal.normalize();

  return normal;
}

// Assume the points are in row-major order
template <typename Point>
[[nodiscard]] tbb::concurrent_vector<PointXYZNTS<double>>
extract_keypoints(const PointCloud<Point> &scan,
                  const KeypointExtractionParams &params, size_t scan_idx) noexcept {
  using T = typename Point::type_t;
  using channel_t = decltype(Point::channel);
  const size_t points_per_sector = scan.num_columns / params.num_sectors;

  // First we validate that all the points are good
  const std::vector<bool> in_range_points = compute_in_range_points(scan, params);
  std::vector<bool> valid_mask = compute_valid_points(scan, params);

  // Next we compute the curvature of each point
  std::vector<Curvature<T>> curvature =
      compute_curvature(scan, valid_mask, params, scan_idx);

  // Next get the planar features
  std::vector<size_t> planar_indices;
  std::vector<size_t> edge_indices;
  for (size_t scan_line_idx = 0; scan_line_idx < scan.num_rows; scan_line_idx++) {
    // Independently detect features in each sector of this scan_line
    for (size_t sector_idx = 0; sector_idx < params.num_sectors; sector_idx++) {
      // Get the point index of the sector start and sector end
      const size_t sector_start_pt =
          (scan_line_idx * scan.num_columns) + (sector_idx * points_per_sector);
      // Special case for end point as we add any reminder points to the last
      // sector
      const size_t sector_end_pt = (sector_idx == params.num_sectors - 1)
                                       ? ((scan_line_idx + 1) * scan.num_columns)
                                       : sector_start_pt + points_per_sector;

      // Sort the points within the sector based on curvature
      std::sort(curvature.begin() + sector_start_pt,
                curvature.begin() + sector_end_pt);

      // Search smallest to largest [i.e. planar features]
      // WARN: Mutates planar_indices + used_points
      extract_planar(sector_start_pt, sector_end_pt, curvature, params,
                     planar_indices, valid_mask);

      extract_edge(sector_start_pt, sector_end_pt, curvature, params, edge_indices,
                   valid_mask);

    } // end sector search
  } // end scan line search

  // Finally extract all normals
  using size_t_iterator = typename std::vector<size_t>::const_iterator;
  tbb::concurrent_vector<PointXYZNTS<double>> result;
  result.reserve(planar_indices.size() + edge_indices.size());
  tbb::parallel_for(tbb::blocked_range<size_t_iterator>{planar_indices.cbegin(),
                                                        planar_indices.cend()},
                    [&](const tbb::blocked_range<size_t_iterator> &range) {
                      for (auto it = range.begin(); it != range.end(); ++it) {
                        const size_t idx = *it;
                        const Point &point = scan.points[idx];
                        std::optional<Eigen::Matrix<T, 3, 1>> normal =
                            compute_normal(idx, scan, params, in_range_points, 0);
                        if (normal.has_value()) {
                          result.emplace_back(
                              static_cast<double>(point.x),
                              static_cast<double>(point.y),
                              static_cast<double>(point.z),
                              static_cast<double>(normal.value().x()),
                              static_cast<double>(normal.value().y()),
                              static_cast<double>(normal.value().z()),
                              static_cast<size_t>(scan_idx), FeatureType::Planar);
                        }
                      }
                    });

  tbb::parallel_for(
      tbb::blocked_range<size_t_iterator>{edge_indices.cbegin(),
                                          edge_indices.cend()},
      [&](const tbb::blocked_range<size_t_iterator> &range) {
        for (auto it = range.begin(); it != range.end(); ++it) {
          const size_t idx = *it;
          const Point &point = scan.points[idx];
          std::optional<Eigen::Matrix<T, 3, 1>> normal =
              compute_edge_direction(idx, scan, params, in_range_points);
          if (normal.has_value()) {
            result.emplace_back(static_cast<double>(point.x),
                                static_cast<double>(point.y),
                                static_cast<double>(point.z),
                                static_cast<double>(normal.value().x()),
                                static_cast<double>(normal.value().y()),
                                static_cast<double>(normal.value().z()),
                                static_cast<size_t>(scan_idx), FeatureType::Edge);
          }
        }
      });

  return result;
}

struct ExtractClassic {
  using Params = KeypointExtractionParams;
  Params params;

  ExtractClassic(const Params &params) : params(params) {}

  template <typename Point>
  std::vector<PointXYZNTS<double>> operator()(const PointCloud<Point> &scan,
                                              size_t scan_idx) const {
    auto keypoints = extract_keypoints(scan, params, scan_idx);
    // copy to a std::vector for return
    return std::vector<PointXYZNTS<double>>(keypoints.begin(), keypoints.end());
  }
};

} // namespace form::classic