cmake_minimum_required(VERSION 3.22)

project(FaceDetection)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

file(GLOB INCLUDE_FILES include/*.hpp include/*.h)
file(GLOB SRC_FILES src/*.cpp src/*.c)

add_executable(FaceDetection main.cpp ${INCLUDE_FILES} ${SRC_FILES})

target_include_directories(FaceDetection PRIVATE include)

target_link_libraries(FaceDetection PRIVATE openvino::runtime ${OpenCV_LIBS})