#include <utils.h>

/*-------------------------------------------------------------------------------*/
// fft-related operations
void fft::calculateDFT2(Mat &src, Mat &dst)
{
    // define mat consists of two mat, one for real values and the other for complex values
    if (src.channels() == 1)
    {
        Mat planes[] = {src, Mat::zeros(src.size(), CV_32F)};
        Mat complexImg;
        merge(planes, 2, complexImg);

        dft(complexImg, complexImg);
        dst = complexImg; // 2-channel mat object, first channel is real part, second one is imagine part
    }
    else if (src.channels() == 2)
    {
        dft(src, dst);
    }
    else
    {
        cout << "Must be 1 or 2 channels, not 3!\n";
    }
}

void fft::resizedDft2(Mat &complexImg, Mat &resizedDftOutput, Size desiredSize)
{
    Size img_sz = Size(complexImg.cols, complexImg.rows);
    if (img_sz.height != desiredSize.height ||
        img_sz.width != desiredSize.width)
    {
        Size min_sz = Size(img_sz.width > desiredSize.width ? desiredSize.width : img_sz.width,
                           img_sz.height > desiredSize.height ? desiredSize.height : img_sz.height);

        float scaling = desiredSize.height * desiredSize.width / (img_sz.width * img_sz.height * 1.0f);

        Size mids = Size(ceil(1.0f * min_sz.height / 2), ceil(1.0f * min_sz.width / 2));
        Size mide = Size(floor((1.0f * min_sz.height - 1) / 2) - 1, floor((1.0f * min_sz.width - 1) / 2) - 1);
        // output
        Mat realOut = Mat::zeros(desiredSize.height, desiredSize.width, CV_32FC1); // rows - cols order
        Mat imagOut = Mat::zeros(desiredSize.height, desiredSize.width, CV_32FC1);

        Mat planes[2];
        split(complexImg, planes);
        Mat realIn = planes[0], imagIn = planes[1];
        //
        Rect roi = Rect(Point(0, 0), Point(mids.width, mids.height));
        Mat temp = realIn(roi);
        temp = scaling * temp;
        temp.copyTo(realOut(roi));

        temp = imagIn(roi);
        temp = scaling * temp;
        temp.copyTo(imagOut(roi));
        //
        Rect roiIn = Rect(Point(img_sz.width - mide.width - 1, 0), Point(img_sz.width, mids.height));
        Rect roiOut = Rect(Point(desiredSize.width - mide.width - 1, 0), Point(desiredSize.width, mids.height));
        temp = realIn(roiIn);
        temp = scaling * temp;
        temp.copyTo(realOut(roiOut));

        temp = imagIn(roiIn);
        temp = scaling * temp;
        temp.copyTo(imagOut(roiOut));
        //
        roiIn = Rect(Point(0, img_sz.height - mide.height - 1), Point(mids.width, img_sz.height));
        roiOut = Rect(Point(0, desiredSize.height - mide.height - 1), Point(mids.width, desiredSize.height));
        temp = realIn(roiIn);
        temp = scaling * temp;
        temp.copyTo(realOut(roiOut));

        temp = imagIn(roiIn);
        temp = scaling * temp;
        temp.copyTo(imagOut(roiOut));
        //
        roiIn = Rect(Point(img_sz.width - mide.width - 1, img_sz.height - mide.height - 1),
                     Point(img_sz.width, img_sz.height));
        roiOut = Rect(Point(desiredSize.width - mide.width - 1, desiredSize.height - mide.height - 1),
                      Point(desiredSize.width, desiredSize.height));
        temp = realIn(roiIn);
        temp = scaling * temp;
        temp.copyTo(realOut(roiOut));

        temp = imagIn(roiIn);
        temp = scaling * temp;
        temp.copyTo(imagOut(roiOut));
        //
        Mat planes_[2] = {realOut, imagOut};
        cv::merge(planes_, 2, resizedDftOutput);
    }
    else
    {
        resizedDftOutput = complexImg;
    }
}

void fft::calculateIDFT2(Mat &src, Mat &dst)
{
    if (src.channels() == 2)
    {
        cv::idft(src, dst, cv::DFT_SCALE | cv::DFT_INVERSE);
    }
    else if (src.channels() == 1)
    {
        Mat planes[2] = {src, Mat::zeros(src.size(), CV_32FC1)};
        Mat complex;
        merge(planes, 2, complex);
        idft(complex, dst, cv::DFT_SCALE | cv::DFT_INVERSE);
    }
};

/*---------------------------------------------------------------------------------*/
void resp::circshift(Mat &src, Mat &dst, Size shiftSize)
{
    Mat tempMat = Mat::zeros(src.size(), src.type());
    int nrow = src.rows;
    int ncol = src.cols;

    int nrow_s = shiftSize.height % nrow;
    if (nrow_s < 0)
        nrow_s += nrow;
    int ncol_s = shiftSize.width % ncol;
    if (ncol_s < 0)
        ncol_s += ncol;
    // shift rows
    Rect roi_in(Point(0, nrow - nrow_s), Point(ncol, nrow));
    Rect roi_out(Point(0, 0), Point(ncol, nrow_s));
    src(roi_in).copyTo(tempMat(roi_out));
    roi_in = Rect(Point(0, 0), Point(ncol, nrow - nrow_s));
    roi_out = Rect(Point(0, nrow_s), Point(ncol, nrow));
    src(roi_in).copyTo(tempMat(roi_out));
    // shift columns
    dst = Mat::zeros(src.size(), src.type());
    roi_in = Rect(Point(ncol - ncol_s, 0), Point(ncol, nrow));
    roi_out = Rect(Point(0, 0), Point(ncol_s, nrow));
    tempMat(roi_in).copyTo(dst(roi_out));
    roi_in = Rect(Point(0, 0), Point(ncol - ncol_s, nrow));
    roi_out = Rect(Point(ncol_s, 0), Point(ncol, nrow));
    tempMat(roi_in).copyTo(dst(roi_out));
}

void resp::shift_sample(Mat &inOutMat, Size shift, Mat kx, Mat ky)
{
    /*
    Shift a sample in the Fourier domain. The shift should be normalized to
    the range [-pi, pi].
    */
    assert(kx.rows == 1 && ky.cols == 1);
    assert(kx.channels() == ky.channels() == 1);
    kx.convertTo(kx, CV_32FC1);
    kx = kx * shift.width;
    ky.convertTo(ky, CV_32FC1);
    ky = ky * shift.height;

    Mat shift_exp_x(kx.size(), CV_32FC2);
    Mat shift_exp_y(ky.size(), CV_32FC2);
    // exp(shift_exp_x)
    vector<Mat> x_planes;
    split(shift_exp_x, x_planes);
    x_planes[0] = MyMat::e_cos(kx);
    x_planes[1] = MyMat::e_sin(kx);
    merge(x_planes, shift_exp_x);
    // exp(shift_exp_y)
    vector<Mat> y_planes;
    split(shift_exp_y, y_planes);
    y_planes[0] = MyMat::e_cos(ky);
    y_planes[1] = MyMat::e_sin(ky);
    merge(y_planes, shift_exp_y);
    // multiply
    inOutMat = MyMat::e_complex_mul(MyMat::e_complex_mul(inOutMat, shift_exp_y), shift_exp_x);
};

Size resp::resp_newton(Mat &xt, Mat &xtf, int newton_iterations, Mat ky, Mat kx, Size use_sz)
{
    assert(xt.channels() == 1 && xtf.channels() == 2);
    float init_max_resp;
    Size maxLoc = MyMat::max_loc(xt, init_max_resp);
    int col = static_cast<int>(maxLoc.width) + 1;
    int row = static_cast<int>(maxLoc.height) + 1;

    int temp = floor((use_sz.height - 1) / 2.0f);
    int trans_row = (row - 1 + temp) % use_sz.height - temp;
    temp = floor((use_sz.width - 1) / 2.0f);
    int trans_col = (col - 1 + temp) % use_sz.width - temp;


    float init_pos_x = 2 * M_PI * trans_col / use_sz.width;
    float init_pos_y = 2 * M_PI * trans_row / use_sz.height;

    float max_pos_x = init_pos_x;
    float max_pos_y = init_pos_y;
    Mat exp_iky = MyMat::exp_complex(ky * (max_pos_y));
    Mat exp_ikx = MyMat::exp_complex(kx * (max_pos_x));

    Mat kx2 = kx.mul(kx);
    kx2 = MyMat::real2Complex(kx2);
    Mat ky2 = ky.mul(ky);
    ky2 = MyMat::real2Complex(ky2);
    int iter = 1;
    Mat ky_exp_ky, kx_exp_kx, y_resp, resp_x, grad_y, grad_x;
    Mat ival_, H_yy, H_xx, H_xy, detH;
    float ival;
    Mat tempmat, temp1;
    while (iter < newton_iterations)
    {
        ky_exp_ky = MyMat::e_mul(exp_iky, ky);

        kx_exp_kx = MyMat::e_mul(exp_ikx, kx);

        y_resp = exp_iky * xtf; 

        resp_x = xtf * exp_ikx;

        grad_y = -MyMat::imag(ky_exp_ky * resp_x);

        grad_x = -MyMat::imag(y_resp * kx_exp_kx);

        tempmat = exp_iky * resp_x;
        vector<Mat> tempPlanes;
        split(tempmat, tempPlanes);
        merge(vector<Mat>{-tempPlanes[1], tempPlanes[0]}, ival_);
        ival = ival_.at<float>(0, 0);

        tempmat = -(MyMat::e_complex_mul(ky2, exp_iky) * resp_x);
        H_yy = MyMat::real(tempmat + ival * Mat::ones(tempmat.size(), tempmat.type()));

        tempmat = -(y_resp * MyMat::e_complex_mul(kx2, exp_ikx));
        H_xx = MyMat::real(tempmat + ival * Mat::ones(tempmat.size(), tempmat.type()));

        H_xy = MyMat::real(-(ky_exp_ky * (xtf * kx_exp_kx)));
                                        
        detH = MyMat::e_mul(H_yy, H_xx) - MyMat::e_mul(H_xy, H_xy);

        temp1 = (MyMat::e_mul(H_yy, grad_y) - MyMat::e_mul(H_xy, grad_x)) / detH.at<float>(0, 0);
        max_pos_y = max_pos_y - temp1.at<float>(0, 0);

        temp1 = MyMat::e_mul(H_yy, grad_x) - MyMat::e_mul(H_xy, grad_y) / detH.at<float>(0, 0);
        max_pos_x = max_pos_x - temp1.at<float>(0, 0);

        exp_iky = MyMat::exp_complex(ky * max_pos_y);
        exp_ikx = MyMat::exp_complex(kx * max_pos_x);

        iter = iter + 1;
    }
    float max_response = 1.0f / (use_sz.height * use_sz.width) * MyMat::real((exp_iky * xtf) * exp_ikx).at<float>(0, 0);

    if (max_response < init_max_resp)
    {
        max_pos_y = init_pos_y;
        max_pos_x = init_pos_x;
    }
    return Size(
        (fmod(max_pos_x + M_PI, 2 * M_PI) - M_PI) / (2 * M_PI) * use_sz.width,
        (fmod(max_pos_y + M_PI, 2 * M_PI) - M_PI) / (2 * M_PI) * use_sz.height
        );
}
/*----------------------------------------------------------------------*/
// operations on Mat data type
void MyMat::make_arr(Mat &inOutArr, int a, int b)
{
    assert(a != b);
    int factor = 1;
    int num = b - a + 1;
    if (a > b)
    {
        factor = -1;
        num = -num;
    }
    if (a == b)
        num = 1;
    inOutArr = Mat::zeros(Size(1, num), CV_32FC1);
    float start = a;
    for (int i = 0; i < num; ++i)
    {
        inOutArr.at<float>(i) = start + i * factor;
    }
}

Mat MyMat::e_plus(const Mat&a, const Mat&b)
{
    assert((a.rows == b.rows && a.cols == b.cols));
}

Mat MyMat::e_mul(const Mat &a, const Mat &b)
{
    /*
    b must be vector-like mat object or b has size equal to a;
    */
    int a_row = a.rows, a_col = a.cols, a_chan = a.channels();
    int b_row = b.rows, b_col = b.cols, b_chan = b.channels();
    assert(b_row == 1 || b_col == 1 || (b_row == a_row && b_col == a_col));
    assert(b_chan <= a_chan);
    Mat ans;
    if (b_row == 1 && b_col == 1)
    {
        ans = a * b.at<float>(0, 0);
        return ans;
    }
    if (b_row == 1)
    {
        Mat temp = Mat::zeros(a.size(), a.type());
        vector<Mat> planes;
        split(temp, planes);
        for (int chanId = 0; chanId < a_chan; ++chanId)
        {
            Rect roi;
            for (int i = 0; i < a_row; ++i)
            {
                roi = Rect(Point(0, i), Point(a_col, i + 1));
                b.copyTo(planes[chanId](roi));
            }
        }
        merge(planes, temp);
        return a.mul(temp);
    }
    else if (b_col == 1)
    {
        Mat temp = Mat::zeros(a.size(), a.type());
        vector<Mat> planes;
        split(temp, planes);
        for (int chanId = 0; chanId < a_chan; ++chanId)
        {
            Rect roi;
            for (int i = 0; i < a_col; ++i)
            {
                roi = Rect(Point(i, 0), Point(i + 1, a_row));

                b.copyTo(planes[chanId](roi));
            }
        }
        merge(planes, temp);
        return a.mul(temp);
    }
    else
    {
        Mat temp = Mat::zeros(a.size(), a.type());
        vector<Mat> planes;
        split(temp, planes);
        for (int chanId = 0; chanId < a_chan; ++chanId)
        {
            Rect roi = Rect(Point(0, 0), Point(a_row, a_col));
            b.copyTo(planes[chanId](roi));
        }
        merge(planes, temp);
        return a.mul(temp);
    }
}

Mat MyMat::e_complex_mul(const Mat &a, const Mat &b)
{
    assert(a.channels() <= 2 && b.channels() <= 2); // assure that 2 input matrices must be complex
    assert(a.rows == b.rows && a.cols == b.cols);   // 2 inputs must have same size
    Mat a_ = a;
    Mat b_ = b;
    Mat ans;
    if (a.channels() == 1 && b.channels() == 1)
    {
        return MyMat::e_mul(a, b);
    }
    if (a.channels() == 1 && b.channels() == 2)
    {
        vector<Mat> b_planes;
        split(b, b_planes);
        b_planes[0] = b_planes[0].mul(a);
        b_planes[1] = b_planes[1].mul(a);
        merge(b_planes, ans);
    }
    if (a.channels() == 2 && b.channels() == 1)
    {
        vector<Mat> a_planes;
        split(a, a_planes);
        a_planes[0] = a_planes[0].mul(b);
        a_planes[1] = a_planes[1].mul(b);
        merge(a_planes, ans);
    }
    if (a.channels() == 2 && b.channels() == 2)
    {
        vector<Mat> a_planes, b_planes;
        split(a, a_planes);
        split(b, b_planes);
        Mat realOut = a_planes[0].mul(b_planes[0]) - b_planes[1].mul(a_planes[1]);
        Mat imagOut = a_planes[1].mul(b_planes[0]) + b_planes[1].mul(a_planes[0]);
        merge(vector<Mat>{realOut, imagOut}, ans);
    }
    return ans;
}

Mat MyMat::e_cos(const Mat &inImg)
{
    assert(inImg.channels() == 1);
    int rows = inImg.rows;
    int cols = inImg.cols;
    Mat ans = Mat::zeros(inImg.size(), CV_32FC1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans.at<float>(i, j) = cos(inImg.at<float>(i, j));
        }
    }
    return ans;
}

Mat MyMat::e_sin(const Mat &inImg)
{
    assert(inImg.channels() == 1);
    int rows = inImg.rows;
    int cols = inImg.cols;
    Mat ans = Mat::zeros(inImg.size(), CV_32FC1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans.at<float>(i, j) = sin(inImg.at<float>(i, j));
        }
    }
    return ans;
}

Mat MyMat::conj(const Mat &in)
{
    assert(in.channels() == 2);
    vector<Mat> planes;
    split(in, planes);
    planes[1] = planes[1] * (-1);
    Mat ans;
    merge(planes, ans);
    return ans;
}

Mat MyMat::e_exp(const Mat &in)
{
    if (in.channels() == 1)
    {
        Mat res;
        cv::exp(in, res);
        return res;
    }
    Mat res(in.size(), CV_32F);
    vector<Mat> in_planes;
    split(in, in_planes);
    vector<Mat> out_planes;
    for (int i = 0; i < in.channels(); i++)
    {
        Mat temp;
        cv::exp(in_planes[i], temp);
        out_planes.push_back(temp);
    }
    merge(out_planes, res);
    return res;
}

Mat MyMat::exp_complex(const Mat &in)
{
    // perform euler formular on 1-channel matrix
    // exp_complex(matrix) = e ^ (1j * matrix)
    Mat cos = MyMat::e_cos(in.clone());
    Mat sin = MyMat::e_sin(in.clone());
    vector<Mat> planes = {cos, sin};
    Mat res;
    merge(planes, res);
    return res;
}

Mat MyMat::real(const Mat &compMat)
{
    assert(compMat.channels() == 2);
    vector<Mat> planes;
    split(compMat, planes);
    return planes[0];
}

Mat MyMat::imag(const Mat &compMat)
{
    assert(compMat.channels() == 2);
    vector<Mat> planes;
    split(compMat, planes);
    return planes[1];
}

Mat MyMat::real2Complex(const Mat& realMat)
{
    assert(realMat.channels() == 1);
    vector<Mat> planes = {realMat, Mat::zeros(realMat.size(), realMat.type())};
    Mat ans;
    merge(planes, ans);
    return ans;
};

Size MyMat::max_loc(Mat in, float &maxVal)
{
    assert(in.channels() == 1);
    in.convertTo(in, CV_32FC1);
    int rows = in.rows;
    int cols = in.cols;
    float max_val = std::numeric_limits<float>::min();
    int max_row_loc;
    int max_col_loc;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (in.at<float>(i, j) > max_val)
            {
                max_val = in.at<float>(i, j);
                max_row_loc = i;
                max_col_loc = j;
            }
        }
    }
    maxVal = max_val;
    return Size(max_col_loc, max_row_loc);
}

/*-----------------------------------------------------------------------------*/
bool updateRefMu(Mat &response_diff, float &refMu, float zeta, float nu)
{
    bool occ;
    response_diff.convertTo(response_diff, CV_32FC1);
    float phi = 0.3f;
    float m = zeta;
    cv::SVD svd;
    Mat w, u, vt;
    svd.compute(response_diff, w, u, vt, SVD::NO_UV);
    cv::MatIterator_<float> max_addr = std::max_element(w.begin<float>(), w.end<float>());
    float eta = *max_addr / 10000.0f;
    if (eta < phi)
    {
        refMu = m / (1 + log(nu * eta + 1));
        return false;
    }
    else
    {
        refMu = 50.0;
        return true;
    }
}
