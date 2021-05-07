#ifndef COLOR_CORRECTION_H
#define COLOR_CORRECTION_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>
#include <set>
#include <iostream>
#include <math.h>

#include "hist.h"

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
extern "C" float cuda_findMax(const float*);
extern "C" float cuda_findMin(const float*);
extern "C" float* cuda_getHistValue(const cv::Mat&);
extern "C" float* cuda_div(const float*, unsigned int, const float);
extern "C" float* cuda_pn(const float*, int, int, unsigned int);
extern "C" void cuda_calc(const cv::Mat&, cv::Mat&, float*, float*);
extern "C" cv::Mat cuda_agcwd(const cv::Mat&, float, bool);
#endif

static
void getCDF(const std::vector<float>& hist, std::vector<float>& cdf)
{
    assert(hist.size() > 0);
    if (hist.size() == 1) {
        cdf.push_back(hist[0]);
        return;
    }

    int i = 1;
    cdf.push_back(hist[0]);
    for (; i < hist.size(); i++) {
        float val = hist[i] + cdf[i-1];
        cdf.push_back(val);
    }
}

static
cv::Mat agcwd(const cv::Mat& src, float alpha, bool truncated_cdf)
{
    assert(src.channels() == 1);

    std::vector<float> hist;
    getHistValue(src, hist, 256);

    std::vector<float> prob;
    float size = src.rows * src.cols;
    for (int i = 0; i < hist.size(); i++) {
        prob.push_back(hist[i]/size);
    }

    std::vector<float> cdf;
    getCDF(hist, cdf);

    float cdf_max = *std::max_element(cdf.begin(), cdf.end());

    for (int i = 0; i < cdf.size(); i++) {
        cdf[i] /=  cdf_max;
    }

    std::set<float> uniqueIntensity;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            uniqueIntensity.insert(src.at<uchar>(i, j));
        }
    }

    float prob_max = *std::max_element(prob.begin(), prob.end());
    float prob_min = *std::min_element(prob.begin(), prob.end());

    std::vector<float> pn_temp;
    for (auto i: prob) {
        pn_temp.push_back((i - prob_min) / (prob_max - prob_min));
    }

    assert(pn_temp.size() == 256);

    float pn_temp_sum = 0;
    for (int i = 0; i < pn_temp.size(); i++) {
        if (pn_temp[i] > 0) {
            pn_temp[i] = prob_max * pow(pn_temp[i], alpha);
        }
        if (pn_temp[i] < 0) {
            pn_temp[i] = prob_max * (-pow((-pn_temp[i]), alpha));
        }
        pn_temp_sum += pn_temp[i];
    }

    for (int i = 0; i < pn_temp.size(); i++) {
        pn_temp[i] /= pn_temp_sum;
    }

    std::vector<float> prob_normalized_wd;
    getCDF(pn_temp, prob_normalized_wd);

    assert(prob_normalized_wd.size() == 256);

    std::vector<float> inverse_cdf;
    if (truncated_cdf) {
        for (int i = 0; i < prob_normalized_wd.size(); i++) {
            inverse_cdf.push_back(
                (0.5 > 1 - prob_normalized_wd[i])? 0.5: 1 - (prob_normalized_wd[i])
            );
        }
    }
    else {
        for (int i = 0; i < prob_normalized_wd.size(); i++) {
            inverse_cdf.push_back(
                1 - prob_normalized_wd[i]
            );
        }
    }

    cv::Mat dst = src.clone();

    for (auto intensity: uniqueIntensity) {

        for (int x = 0; x < src.rows; x++) {
            for (int y = 0; y < src.cols; y++) {
               if (src.at<uchar>(x, y) == intensity) {
                   dst.at<uchar>(x, y) = round(255 * pow((intensity/255), inverse_cdf[intensity]));
               }
            }
        }

    }

    return dst;
}

static
cv::Mat processDimmed(const cv::Mat& src)
{
    assert(src.channels() == 1);

#ifdef CUDA
    cv::Mat dst = cuda_agcwd(src, 0.75, true);
#else
    cv::Mat dst = agcwd(src, 0.75, true);
#endif

    return dst;
}

static
cv::Mat processBright(const cv::Mat& src)
{
    assert(src.channels() == 1);

    cv::Mat negative;
    cv::bitwise_not(src, negative);
#ifdef CUDA
    cv::Mat agcwd_img = cuda_agcwd(negative, 0.25, false);
#else
    cv::Mat agcwd_img = agcwd(negative, 0.25, false);
#endif
    cv::Mat reversed;
    cv::bitwise_not(agcwd_img, reversed);

    return reversed;
}

/**
 * @ref https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py
 */
static
void colorCorrection(const cv::Mat& src, cv::Mat& dst)
{
    assert(src.channels() == 3);

    cv::Mat YCrCb;
    cv::cvtColor(src, YCrCb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(YCrCb, channels);

    float threshold = 0.3;
    float expIntensity = 112.0;
    float meanIntensity = 0.0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            meanIntensity += channels[0].at<uchar>(i, j);
        }
    }
    meanIntensity /= src.rows * src.cols;

    float t = (meanIntensity - expIntensity) / expIntensity;

    if (t < -threshold) {
        cv::Mat new_Y = processDimmed(channels[0]);
        channels[0] = new_Y;
        cv::Mat new_YCrCb;
        cv::merge(channels, new_YCrCb);
        cv::cvtColor(new_YCrCb, dst, cv::COLOR_YCrCb2BGR);
    }
    else if (t > threshold) {
        cv::Mat new_Y = processBright(channels[0]);
        channels[0] = new_Y;
        cv::Mat new_YCrCb;
        cv::merge(channels, new_YCrCb);
        cv::cvtColor(new_YCrCb, dst, cv::COLOR_YCrCb2BGR);
    }
    else {
        dst = src.clone();
    }
}

#endif
