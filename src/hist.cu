#include "hist.h"

extern "C"
float* cuda_getHistValue(const cv::Mat&src)
{
    assert(src.channels() == 1);
    assert(src.type() == CV_8U);

    cv::Mat tmp = src.clone();
    cv::cuda::GpuMat d_img;
    cv::cuda::GpuMat d_hist;
    d_img.upload(tmp);
    cv::cuda::calcHist(d_img, d_hist);
    cv::Mat h_hist; // int type
    d_hist.download(h_hist);

    // convert to float type
    float* result = (float*)malloc(256 * sizeof(float*));
    for (int i = 0; i < 256; i++) {
        result[i] = (float)h_hist.at<int>(0, i);
    }

    return result;
}