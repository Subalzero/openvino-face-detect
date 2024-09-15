cmake_minimum_required(VERSION 3.22)

project(FaceDetection)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

file(GLOB INCLUDE_FILES include/*.hpp include/*.h)
file(GLOB SRC_FILES src/*.cpp src/*.c)

add_executable(FaceDetection main.cpp ${INCLUDE_FILES} ${SRC_FILES})

target_include_directories(FaceDetection PRIVATE include)

target_link_libraries(FaceDetection PRIVATE openvino::runtime ${OpenCV_LIBS})

add_custom_command(
    TARGET FaceDetection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    ${CMAKE_CURRENT_BINARY_DIR}/models
)

if (WIN32)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_custom_command(
                TARGET FaceDetection POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/thirdparty/vcpkg/packages/openvino_x64-windows/debug/bin
                ${CMAKE_CURRENT_BINARY_DIR}
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_custom_command(
                TARGET FaceDetection POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/thirdparty/vcpkg/packages/openvino_x64-windows/bin
                ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
endif()