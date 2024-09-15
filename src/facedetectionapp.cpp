#include "facedetectionapp.h"

FaceDetectionApp::FaceDetectionApp(int argc, char **argv)
{
#if _WIN32
    capture = cv::VideoCapture(0, cv::CAP_MSMF);
#else
    capture = cv::VideoCapture(0, cv::CAP_V4L2);
#endif

    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    auto devices = OpenVINORunner::get_available_devices();

    face_detection = OpenVINORunner("models/face-detection-adas-0001.xml", "CPU");
}

FaceDetectionApp::~FaceDetectionApp()
{
    if (capture.isOpened())
    {
        capture.release();
    }
    cv::destroyAllWindows();
}

FaceDetectionApp::FaceDetectionApp(FaceDetectionApp &&temp) noexcept
{
    capture = std::move(temp.capture);

    capture.release();
}

FaceDetectionApp &FaceDetectionApp::operator=(FaceDetectionApp &&temp) noexcept
{
    capture = std::move(temp.capture);

    capture.release();

    return *this;
}

int FaceDetectionApp::exec()
{
    while (capture.isOpened())
    {
        cv::Mat frame;
        int ret = capture.read(frame);

        if (ret)
        {
            cv::namedWindow("Face Detection", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            cv::imshow("Face Detection", frame);
            if (cv::waitKey(1) == 27)
            {
                break;
            }
        }
    }
    return 0;
}