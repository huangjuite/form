#include "form/point/factor.hpp"

#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include <cstddef>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Values.h>

#include <random>

#include "gtest/gtest.h"

using gtsam::symbol_shorthand::X;

#define EXPECT_MATRICES_EQ(M_got, actual)                                           \
  EXPECT_TRUE(M_got.isApprox(actual, 1e-5)) << "  got:\n"                           \
                                            << M_got << "\actual:\n"                \
                                            << actual

#define EXPECT_ZERO(v) EXPECT_TRUE(v.isZero(1e-4)) << " got is not zero:\n" << v

using namespace form::point;

std::vector<PointConstraint> make_point_constraints(gtsam::Pose3 Ti, gtsam::Pose3 Tj,
                                                    int n) {
  std::mt19937 generator(123);
  std::uniform_real_distribution<float> distr(0.0, 10.0);

  std::vector<PointConstraint> constraints;
  // Make x wall constraints
  for (int i = 0; i < n; ++i) {
    gtsam::Vector3 p_i(0.0, distr(generator), distr(generator));
    constraints.push_back(PointConstraint(Ti.transformTo(p_i), Tj.transformTo(p_i)));
  }

  // Make y wall constraints
  for (int i = 0; i < n; ++i) {
    gtsam::Vector3 p_i(distr(generator), 0.0, distr(generator));
    constraints.push_back(PointConstraint(Ti.transformTo(p_i), Tj.transformTo(p_i)));
  }

  // Make z wall constraints
  for (int i = 0; i < n; ++i) {
    gtsam::Vector3 p_i(distr(generator), distr(generator), 0.0);
    constraints.push_back(PointConstraint(Ti.transformTo(p_i), Tj.transformTo(p_i)));
  }

  return constraints;
}

// Verify old PointConstraint jacobians match new PointFactor ones
TEST(Point, Residuals) {
  // Setup states
  gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
  gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});

  // Setup pf
  std::vector<PointConstraint> constraints = {
      PointConstraint(gtsam::Point3(0.1, 0.2, 0.4), gtsam::Point3(0.5, 0.6, 0.7)),
      PointConstraint(gtsam::Point3(0.1, 0.2, 0.4), gtsam::Point3(0.5, 0.6, 0.7)),
  };

  // got
  gtsam::Vector actual;
  gtsam::Matrix H1_exp, H2_exp, H1_exp_final, H2_exp_final;
  actual.resize(3 * constraints.size());
  H1_exp_final.resize(3 * constraints.size(), 6);
  H2_exp_final.resize(3 * constraints.size(), 6);
  for (size_t i = 0; i < constraints.size(); ++i) {
    actual.segment<3>(3 * i) = constraints[i].evaluateError(x0, x1, H1_exp, H2_exp);
    H1_exp_final.block<3, 6>(3 * i, 0) = H1_exp;
    H2_exp_final.block<3, 6>(3 * i, 0) = H2_exp;
  }

  // Got
  gtsam::Matrix H1, H2;
  auto pf = PointFactor(X(0), X(1), constraints, 1.0);
  gtsam::Vector got = pf.evaluateError(x0, x1, H1, H2);

  EXPECT_MATRICES_EQ(actual, got);
  EXPECT_MATRICES_EQ(H1_exp_final, H1);
  EXPECT_MATRICES_EQ(H2_exp_final, H2);
}

TEST(Point, FactorJacobians) {
  // Setup states
  gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
  gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});

  // Setup pf
  auto pc =
      make_point_constraints(gtsam::Pose3::Identity(), gtsam::Pose3::Identity(), 5);
  auto pf = PointFactor(X(0), X(1), pc, 1.0);

  // Setup lambda
  std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Pose3 &)>
      errorComputer = [pf](const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j) {
        return pf.evaluateError(pose_i, pose_j);
      };

  gtsam::Matrix H1, H2, H1_num, H2_num;
  gtsam::Vector e = pf.evaluateError(x0, x1, H1, H2);

  H1_num = gtsam::numericalDerivative21(errorComputer, x0, x1);
  H2_num = gtsam::numericalDerivative22(errorComputer, x0, x1);

  // std::cout << H1 << "\n";
  // std::cout << H1_num << "\n\n";
  // std::cout << H2 << "\n";
  // std::cout << H2_num << "\n\n";

  EXPECT_MATRICES_EQ(H1, H1_num);
  EXPECT_MATRICES_EQ(H2, H2_num);
}

TEST(Point, Optimize) {
  // Setup states
  gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3), {1, 2, 3});
  gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.4, 0.5, 0.6), {4, 5, 6});

  // Setup pf
  auto pc = make_point_constraints(x0, x1, 5);
  auto pf = PointFactor(X(0), X(1), pc, 1.0);

  // prior
  auto prior = gtsam::PriorFactor<gtsam::Pose3>(
      X(0), x0, gtsam::noiseModel::Isotropic::Sigma(6, 1e-5));

  // Graph and values
  gtsam::NonlinearFactorGraph graph;
  graph.push_back(prior);
  graph.push_back(pf);

  gtsam::Values initial;
  initial.insert(X(0), gtsam::Pose3::Identity());
  initial.insert(X(1), gtsam::Pose3::Identity());

  // Optimize
  gtsam::LevenbergMarquardtParams params;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
  gtsam::Values results = optimizer.optimize();
  gtsam::Pose3 x0_final = results.at<gtsam::Pose3>(X(0));
  gtsam::Pose3 x1_final = results.at<gtsam::Pose3>(X(1));

  // std::cout << "X0\n" << x0 << "\n" << x0_final << "\n";
  // std::cout << "X1\n" << x1 << "\n" << x1_final << "\n";

  EXPECT_ZERO(pf.evaluateError(x0_final, x1_final));
  EXPECT_ZERO(x0.localCoordinates(x0_final));
  EXPECT_ZERO(x1.localCoordinates(x1_final));
}
