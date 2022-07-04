#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

namespace fft
{
    void calculateDFT2(Mat &src, Mat &dst);

    void resizedDft2(Mat &complexImg, Mat &resizedDftOutput, Size desiredSize);

    void calculateIDFT2(Mat &src, Mat &dst);
}

namespace resp
{
    void circshift( Mat& src, Mat& dst, Size shift_size);

    void respDiff();
}

namespace MyMat 
{
    void make_arr(Mat &inOutArr, int a, int b);
}

bool updateRefMu(Mat &response_diff, float &refMu, float zeta=13.0f, float nu=0.3f);