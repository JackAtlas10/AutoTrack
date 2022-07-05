#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <iostream>
// #include <math.h>
using namespace std;
using namespace cv;

namespace fft
{
    void calculateDFT2(Mat &src, Mat &dst);

    void resizedDft2(Mat &complexImg, Mat &resizedDftOutput, Size desiredSize);

    void calculateIDFT2(Mat &src, Mat &dst);
}

namespace resp
{
    void circshift( Mat& src, Mat& dst, Size shift_size);

    void shift_sample(Mat &inOutMat, Size shift, Mat kx, Mat ky);

    void resp_newton(Mat &xt, Mat &xtf, Size displacement, int iterations, Mat kx, Mat ky, Size use_sz);

    void respDiff();
}

namespace MyMat 
{
    void make_arr(Mat &inOutArr, int a, int b);

    Mat e_mul(const Mat&a, const Mat&b);

    Mat e_cos(const Mat &in);

    Mat e_sin(const Mat &in);

    Mat e_complex_mul(const Mat&a, const Mat&b);

    Mat conj(const Mat& in);
}

bool updateRefMu(Mat &response_diff, float &refMu, float zeta=13.0f, float nu=0.3f);