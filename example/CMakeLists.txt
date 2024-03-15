project(mathematical_robotics CXX)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20) 
add_compile_options(-O3)

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(${CERES_INCLUDE_DIR})
include_directories(../utilities)

set(libs 
    Eigen3::Eigen
    Matplot++::matplot
)

add_executable(growing_neural_gas "growing_neural_gas.cpp")
target_link_libraries(growing_neural_gas 
    ${libs}
)