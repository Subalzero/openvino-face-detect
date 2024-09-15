#ifndef FACE_DETECTION_APP_H
#define FACE_DETECTION_APP_H

#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

#include "openvinorunner.h"

class FaceDetectionApp
{
public:
    FaceDetectionApp() = delete;

    FaceDetectionApp(int argc, char **argv);
    ~FaceDetectionApp();

    FaceDetectionApp(const FaceDetectionApp &copy) = delete;
    FaceDetectionApp &operator=(const FaceDetectionApp &rhs) = delete;

    FaceDetectionApp(FaceDetectionApp &&temp) noexcept;
    FaceDetectionApp &operator=(FaceDetectionApp &&temp) noexcept;

    int exec();

private:
    cv::VideoCapture capture;

    OpenVINORunner face_detection;
};

#endif // FACE_DETECTION_APP_H