#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <limits>
#include <functional>
#include <memory>

#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <open3d/Open3D.h>

#include "growing_neural_gas.hpp"
#include "math_utils.hpp"
#include "plot_helper.hpp"

namespace plt = matplot;

constexpr size_t D = 9;

int main()
{
    const double point_cloud_range = 1.0;

    std::mt19937 engine(0);
    std::uniform_real_distribution<> pos_rand(-point_cloud_range, point_cloud_range);
    std::uniform_real_distribution<> error_rand(-0.1, 0.1);

    constexpr size_t point_cloud_num = 6000;

    std::vector<Eigen::Vector3d> plane_x;
    std::vector<Eigen::Vector3d> plane_y;
    std::vector<Eigen::Vector3d> plane_z;

    plane_x.reserve(point_cloud_num / 3);
    plane_y.reserve(point_cloud_num / 3);
    plane_z.reserve(point_cloud_num / 3);
    for (size_t i = 0; i < point_cloud_num / 3; i++) {
        plane_z.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine), pos_rand(engine) + error_rand(engine), 0.0));
        plane_y.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine), point_cloud_range, pos_rand(engine) + error_rand(engine) + point_cloud_range));
        plane_x.push_back(Eigen::Vector3d(point_cloud_range, pos_rand(engine) + error_rand(engine), pos_rand(engine) + error_rand(engine) + point_cloud_range));

        // plane_z.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine) - 2, pos_rand(engine) + error_rand(engine) + 1, 0.0));
        // plane_y.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine) + 3, 0.0, pos_rand(engine) + error_rand(engine) - 1));
        // plane_x.push_back(Eigen::Vector3d(0.0, pos_rand(engine) + error_rand(engine) + 1, pos_rand(engine) + error_rand(engine) + 3));
    }

    auto plane_x_pcd = std::make_shared<open3d::geometry::PointCloud>(plane_x);
    plane_x_pcd->PaintUniformColor({1.0, 0.0, 0.0});
    auto plane_y_pcd = std::make_shared<open3d::geometry::PointCloud>(plane_y);
    plane_y_pcd->PaintUniformColor({0.0, 1.0, 0.0});
    auto plane_z_pcd = std::make_shared<open3d::geometry::PointCloud>(plane_z);
    plane_z_pcd->PaintUniformColor({0.0, 0.0, 1.0});
    // open3d::visualization::DrawGeometries({plane_x_pcd, plane_y_pcd, plane_z_pcd});

    // auto all_pcd = std::make_shared<open3d::geometry::PointCloud>(*plane_x_pcd + *plane_y_pcd + *plane_z_pcd);
    // all_pcd->EstimateNormals();
    // open3d::visualization::DrawGeometries({all_pcd}, "Point Cloud", 640, 480, 50, 50, true);

    auto all_pcd = open3d::io::CreatePointCloudFromFile("dataset/fragment.ply");
    // open3d::visualization::DrawGeometries({all_pcd}, "Point Cloud", 640, 480, 50, 50, false);
    // all_pcd = all_pcd->FarthestPointDownSample(point_cloud_num);
    // all_pcd->EstimateNormals();
    // open3d::visualization::DrawGeometries({all_pcd}, "Point Cloud", 640, 480, 50, 50, false);

    std::uniform_int_distribution<> sample_rand(0, all_pcd->points_.size() - 1);
    int rand1 = sample_rand(engine);
    int rand2 = sample_rand(engine);
    while (rand1 == rand2) {
        rand2 = sample_rand(engine);
    }

    const double normals_weight = 0.2;
    const double color_weight = 0.2;

    Eigen::Matrix<double, D, 1> rand1_vec;
    rand1_vec << all_pcd->points_[rand1], normals_weight * all_pcd->normals_[rand1], color_weight * all_pcd->colors_[rand1];
    Eigen::Matrix<double, D, 1> rand2_vec;
    rand2_vec << all_pcd->points_[rand2], normals_weight * all_pcd->normals_[rand2], color_weight * all_pcd->colors_[rand2];
    GrowingNeuralGas<D> gng({rand1_vec, rand2_vec});

    // plt::hold(plt::on);
    // plot_helper::plot_point_cloud(plane_x, "blue");
    // plot_helper::plot_point_cloud(plane_y, "red");
    // plot_helper::plot_point_cloud(plane_z, "green");
    // plt::hold(plt::off);
    // plt::save("img/start.png");
    // plt::cla();

    std::cout << "Start GNG" << std::endl;
    for (size_t i = 0; i < 1000000; i++) {
        int rand = sample_rand(engine);
        Eigen::Matrix<double, D, 1> p;
        p << all_pcd->points_[rand], normals_weight * all_pcd->normals_[rand], color_weight * all_pcd->colors_[rand];
        gng.update(p, 0.2, 0.006, 50, 100, 0.5, 0.995);

        // if (i % 100 == 0) {
        //     plt::hold(plt::on);
        //     plot_helper::plot_point_cloud(plane_x, "blue");
        //     plot_helper::plot_point_cloud(plane_y, "red");
        //     plot_helper::plot_point_cloud(plane_z, "green");
        //     gng.plot();
        //     plt::hold(plt::off);
        //     plt::save("img/" + std::to_string(i) + ".png");
        //     plt::cla();
        // }
    }
    std::cout << "End GNG" << std::endl;
    // all_pcd->PaintUniformColor({0.0, 0.0, 0.0});
    open3d::visualization::DrawGeometries({gng.to_open3d_point_cloud(), gng.to_open3d_line_set()});
    // plt::hold(plt::on);
    // plot_helper::plot_point_cloud(plane_x, "blue");
    // plot_helper::plot_point_cloud(plane_y, "red");
    // plot_helper::plot_point_cloud(plane_z, "green");
    // gng.plot();
    // plt::hold(plt::off);
    // plt::save("img/end.png");
    // plt::show();
}
