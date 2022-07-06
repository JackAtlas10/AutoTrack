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

    Size resp_newton(Mat &xt, Mat &xtf, int iterations, Mat kx, Mat ky, Size use_sz);

    

    void respDiff();
}

namespace MyMat 
{
    void make_arr(Mat &inOutArr, int a, int b);

    Mat e_complex_mul(const Mat&a, const Mat&b); // element-wise complex multiplication

    Mat e_mul(const Mat&a, const Mat&b);

    Mat e_cos(const Mat &in);

    Mat e_sin(const Mat &in);

    Mat conj(const Mat& in);

    Mat e_exp(const Mat& in);

    Mat exp_complex(const Mat& in);

    Mat real(const Mat& compMat);

    Mat imag(const Mat& compMat);

    

    Size max_loc(Mat in);
}

bool updateRefMu(Mat &response_diff, float &refMu, float zeta=13.0f, float nu=0.3f);