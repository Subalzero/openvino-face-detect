cmake_minimum_required(VERSION 3.22)

project(FaceDetection)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(FaceDetection main.cpp)

target_link_libraries(FaceDetection PRIVATE openvino::runtime ${OpenCV_LIBS})