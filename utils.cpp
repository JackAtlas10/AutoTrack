#include <utils.h>

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
    assert(kx.cols == 1 && ky.rows == 1);
    kx.convertTo(kx, CV_32FC1);
    kx = kx * shift.width;
    ky.convertTo(ky, CV_32FC1);
    ky = ky * shift.height;

    Mat shift_exp_x(kx.size(), CV_32FC2);
    Mat shift_exp_y(ky.size(), CV_32FC2);
    vector<Mat> x_planes;
    split(shift_exp_x, x_planes);
    vector<Mat> y_planes;
    split(shift_exp_y, y_planes);
};

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

Mat MyMat::e_mul(const Mat &a, const Mat &b)
{
    /*
    b must be vector-like mat object or b has size equal to a;
    */
    int a_row = a.rows, a_col = a.cols, a_chan = a.channels();
    int b_row = b.rows, b_col = b.cols, b_chan = b.channels();
    assert(b_row == 1 || b_col == 1 || (b_row == a_row && b_col == a_col));
    assert(b_row != 1 || b_col != 1);
    assert(b_chan <= a_chan);
    Mat ans;
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
