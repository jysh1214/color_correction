#ifndef HIST_H
#define HIST_H

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp> // cv::cuda::PtrStepSz
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
extern "C" float* cuda_getHistValue(const cv::Mat&src);
#endif

static
void getHistValue(const cv::Mat& src, std::vector<float>& histValue, const int size)
{
    assert(src.type() == CV_8U);
    assert(src.channels() == 1);

    const int channels[1] = {0};
    int histSize[1] = {size};

    float hrange[2] = {0, (float)size};
    const float* range[1] = {hrange};
    cv::Mat histMat;
    cv::calcHist(&src, 1, channels, cv::Mat(), histMat, 1, histSize, range);

    for (int i = 0; i < size; i++)
        histValue.push_back(histMat.at<float>(i, 0));
}

#endif