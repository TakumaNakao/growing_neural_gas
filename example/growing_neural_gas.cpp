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

#include "growing_neural_gas.hpp"
#include "math_utils.hpp"
#include "plot_helper.hpp"

namespace plt = matplot;

int main()
{
    const double point_cloud_range = 1.0;

    std::mt19937 engine(0);
    std::uniform_real_distribution<> pos_rand(-point_cloud_range, point_cloud_range);
    std::uniform_real_distribution<> error_rand(-0.1, 0.1);

    constexpr size_t point_cloud_num = 900;

    std::vector<Eigen::Vector3d> point_cloud;
    std::vector<Eigen::Vector3d> plane_x;
    std::vector<Eigen::Vector3d> plane_y;
    std::vector<Eigen::Vector3d> plane_z;

    point_cloud.reserve(point_cloud_num);
    plane_x.reserve(point_cloud_num / 3);
    plane_y.reserve(point_cloud_num / 3);
    plane_z.reserve(point_cloud_num / 3);
    for (size_t i = 0; i < point_cloud_num / 3; i++) {
        // plane_z.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine), pos_rand(engine) + error_rand(engine), 0.0));
        // plane_y.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine), point_cloud_range, pos_rand(engine) + error_rand(engine) + point_cloud_range));
        // plane_x.push_back(Eigen::Vector3d(point_cloud_range, pos_rand(engine) + error_rand(engine), pos_rand(engine) + error_rand(engine) + point_cloud_range));

        plane_z.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine) - 2, pos_rand(engine) + error_rand(engine) + 1, 0.0));
        plane_y.push_back(Eigen::Vector3d(pos_rand(engine) + error_rand(engine) + 3, 0.0, pos_rand(engine) + error_rand(engine) - 1));
        plane_x.push_back(Eigen::Vector3d(0.0, pos_rand(engine) + error_rand(engine) + 1, pos_rand(engine) + error_rand(engine) + 3));
    }
    std::copy(plane_x.begin(), plane_x.end(), std::back_inserter(point_cloud));
    std::copy(plane_y.begin(), plane_y.end(), std::back_inserter(point_cloud));
    std::copy(plane_z.begin(), plane_z.end(), std::back_inserter(point_cloud));

    std::vector<Eigen::Vector3d> result;
    std::sample(point_cloud.begin(), point_cloud.end(), std::back_inserter(result), 2, engine);
    GrowingNeuralGas gng({result[0], result[1]});

    plt::hold(plt::on);
    plot_helper::plot_point_cloud(plane_x, "blue");
    plot_helper::plot_point_cloud(plane_y, "red");
    plot_helper::plot_point_cloud(plane_z, "green");
    plt::hold(plt::off);
    plt::save("img/start.png");
    plt::cla();

    for (size_t i = 0; i < 10000; i++) {
        std::vector<Eigen::Vector3d> result;
        std::sample(point_cloud.begin(), point_cloud.end(), std::back_inserter(result), 1, engine);
        gng.update(result[0], 0.2, 0.006, 50, 100, 0.5, 0.995);

        if (i % 100 == 0) {
            plt::hold(plt::on);
            plot_helper::plot_point_cloud(plane_x, "blue");
            plot_helper::plot_point_cloud(plane_y, "red");
            plot_helper::plot_point_cloud(plane_z, "green");
            gng.plot();
            plt::hold(plt::off);
            plt::save("img/" + std::to_string(i) + ".png");
            plt::cla();
        }
    }
    plt::hold(plt::on);
    plot_helper::plot_point_cloud(plane_x, "blue");
    plot_helper::plot_point_cloud(plane_y, "red");
    plot_helper::plot_point_cloud(plane_z, "green");
    gng.plot();
    plt::hold(plt::off);
    plt::save("img/end.png");
    plt::cla();
}
