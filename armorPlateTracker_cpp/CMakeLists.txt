cmake_minimum_required(VERSION 3.27)
project(armorPlateTracker_cpp)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(armorPlateTracker-cpp main.cpp)
target_link_libraries(armorPlateTracker-cpp ${OpenCV_LIBS})
