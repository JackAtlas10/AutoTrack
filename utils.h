#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <iostream>
// #include <math.h>
using namespace std;
using namespace cv;

namespace fft
{
    void calculateDFT2(cv::Mat &src, cv::Mat &dst);

    void resizedDft2(cv::Mat &complexImg, cv::Mat &resizedDftOutput, Size desiredSize);

    void calculateIDFT2(cv::Mat &src, cv::Mat &dst);
}

namespace resp
{
    void circshift( cv::Mat& src, cv::Mat& dst, Size shift_size);

    void shift_sample(cv::Mat &inOutMat, Size shift, cv::Mat kx, cv::Mat ky);

    void respDiff();
}

namespace MyMat 
{
    void make_arr(cv::Mat &inOutArr, int a, int b);

    cv::Mat e_mul(const cv::Mat&a, const cv::Mat&b);

    Mat e_cos(const Mat &in);

    Mat e_sin(const Mat &in);
}

bool updateRefMu(cv::Mat &response_diff, float &refMu, float zeta=13.0f, float nu=0.3f);