#include "color_correction.h"

#define BLOCK_SIZE 8

/**
 * @ref https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu
 */
__device__ 
void atomicFloatMax(float* const address, const float value)
{
    if (*address >= value) {
        return;
    }
  
    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;
  
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) {
            break;
        }
  
        old = atomicCAS(addressAsI, assumed, __float_as_int(value));
    } while (assumed != old);
}

__device__ 
void atomicFloatMin(float* const address, const float value)
{
    if (*address <= value) {
        return;
    }
   
    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;
   
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) {
            break;
        }
   
        old = atomicCAS(addressAsI, assumed, __float_as_int(value));
    } while (assumed != old);
}

/**
 * @ref https://www.youtube.com/watch?v=ZpMMcoCe4Yg
 * @ref https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu
 */
__global__ 
void find_max_kernel(const float* __restrict__ array, float* max, unsigned int arraySize)
{
    __shared__ float sharedMax;

    if (threadIdx.x == 0) {
        sharedMax = array[0];
    }

    __syncthreads();

    float localMax = sharedMax;

    for (unsigned int i = threadIdx.x; i < arraySize; i += blockDim.x) {
        localMax = (array[i] > localMax)? array[i]: localMax;
    }

    atomicFloatMax(&sharedMax, localMax);
    __syncthreads();

    if (threadIdx.x == 0) {
        *max = sharedMax;
    }
}

__global__ 
void find_min_kernel(const float* __restrict__ array, float* min, unsigned int arraySize)
{
    __shared__ float sharedMin;

    if (threadIdx.x == 0) {
        sharedMin = array[0];
    }

    __syncthreads();

    float localMin = sharedMin;

    for (unsigned int i = threadIdx.x; i < arraySize; i += blockDim.x) {
        localMin = (array[i] < localMin)? array[i]: localMin;
    }

    atomicFloatMin(&sharedMin, localMin);
    __syncthreads();

    if (threadIdx.x == 0) {
        *min = sharedMin;
    }
}

__global__ 
void div_kernel(const float* __restrict__ array, unsigned int size, float target, float* result)
{
    unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x < size) {
        result[id_x] = array[id_x] / target;
    }
}

__global__ 
void pn_kernel(const float* __restrict__ prob, float prob_max, float prob_min, unsigned int size, float* pn)
{
    unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x < size) {
        pn[id_x] = (prob[id_x] - prob_min) / (prob_max - prob_min);
    }
}

__global__ 
void calc_kernel(cv::cuda::PtrStepSz<uchar1> d_src,
    cv::cuda::PtrStepSz<uchar1> d_dst, 
    const float* __restrict__ d_intensity, 
    const float* __restrict__ d_inverse_cdf)
{
    unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (id_x < d_src.rows && id_y < d_src.cols && id_z < 256) {
        if (d_src(id_x, id_y).x == id_z && d_intensity[id_z] == 1) {
            d_dst(id_x, id_y).x = round(255 * pow(((float)id_z / 255), d_inverse_cdf[id_z]));
        }
    }
}

/**
 * @param h_array - fixed 256 size array
 * @brief to find max element in the array
 */
extern "C"
float cuda_findMax(const float* h_array)
{
    unsigned int size = 256;
    float* d_array;
    float* d_max;
    int* d_mutex;
    cudaMalloc((void**)&d_array, size * sizeof(float));
    cudaMalloc((void**)&d_max, 1 * sizeof(float));
    cudaMalloc((void**)&d_mutex, 1 * sizeof(int));

    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_max, 0, sizeof(float));
    cudaMemset(d_mutex, 0, sizeof(float));

    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    find_max_kernel<<<grid, block>>>(d_array, d_max, size);
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "find_max_kernel fault\n";
    }

    float* h_max = (float*)malloc(1 * sizeof(float));
    cudaMemcpy(h_max, d_max, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    float result = *h_max;
    free(h_max);

    cudaFree(d_array);
    cudaFree(d_max);
    cudaFree(d_mutex);

    return result;
}

extern "C"
float cuda_findMin(const float* h_array)
{
    unsigned int size = 256;
    float* d_array;
    float* d_min;
    int* d_mutex;
    cudaMalloc((void**)&d_array, size * sizeof(float));
    cudaMalloc((void**)&d_min, 1 * sizeof(float));
    cudaMalloc((void**)&d_mutex, 1 * sizeof(int));

    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_min, 0, sizeof(float));
    cudaMemset(d_mutex, 0, sizeof(float));

    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    find_min_kernel<<<grid, block>>>(d_array, d_min, size);
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "find_min_kernel fault\n";
    }

    float* h_min = (float*)malloc(1 * sizeof(float));
    cudaMemcpy(h_min, d_min, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    float result = *h_min;
    free(h_min);

    cudaFree(d_array);
    cudaFree(d_min);
    cudaFree(d_mutex);

    return result;
}

extern "C"
float* cuda_div(const float* h_array, unsigned int size, const float target)
{
    float* d_array;
    cudaMalloc((void**)&d_array, size * sizeof(float));
    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

    float* h_result = (float*)malloc(size * sizeof(float));
    float* d_result;
    cudaMalloc((void**)&d_result, size * sizeof(float));

    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    div_kernel<<<grid, block>>>(d_array, size, target, d_result);
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "div_kernel fault\n";
    }

    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_result);

    return h_result;
}

extern "C"
float* cuda_pn(const float* h_prob, int prob_max, int prob_min, unsigned int size)
{
    float* h_pn = (float*)malloc(size * sizeof(float));
    float* d_prob;
    cudaMalloc((void**)&d_prob, size * sizeof(float));
    cudaMemcpy(d_prob, h_prob, size * sizeof(float), cudaMemcpyHostToDevice);
    float* d_pn;
    cudaMalloc((void**)&d_pn, size * sizeof(float));

    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    pn_kernel<<<grid, block>>>(d_prob, prob_max, prob_min, size, d_pn);
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "pn_kernel fault\n";
    }

    cudaMemcpy(h_pn, d_pn, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_prob);
    cudaFree(d_pn);

    return h_pn;
}

extern "C"
void cuda_calc(const cv::Mat& h_src, cv::Mat& h_dst, float* h_intensity, float* h_inverse_cdf)
{
    h_dst = h_src.clone();
    cv::cuda::GpuMat d_src;
    cv::cuda::GpuMat d_dst;

    d_src.upload(h_src);
    d_dst.upload(h_dst);

    float* d_intensity;
    cudaMalloc((void**)&d_intensity, 256 * sizeof(float));
    cudaMemcpy(d_intensity, h_intensity, 256 * sizeof(float), cudaMemcpyHostToDevice);

    float* d_inverse_cdf;
    cudaMalloc((void**)&d_inverse_cdf, 256 * sizeof(float));
    cudaMemcpy(d_inverse_cdf, h_inverse_cdf, 256 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    unsigned int grid_x = (h_src.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_y = (h_src.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_z = (256 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, grid_y, grid_z);
    calc_kernel<<<grid, block>>>(d_src, d_dst, d_intensity, d_inverse_cdf);
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "calc_kernel fault\n";
    }

    d_dst.download(h_dst);

    cudaFree(d_intensity);
    cudaFree(d_inverse_cdf);
}

extern "C"
cv::Mat cuda_agcwd(const cv::Mat& src, float alpha, bool truncated_cdf)
{
    assert(src.channels() == 1);

    float* hist = cuda_getHistValue(src);

    float* prob = (float*)malloc(256 * sizeof(float));
    float size = src.rows * src.cols;

    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
        prob[i] = hist[i] / size;
    }

    float* cdf = (float*)malloc(256 * sizeof(float));
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + hist[i];
    }

    float cdf_max = cuda_findMax(cdf);
    float* norm_cdf = cuda_div(cdf, 256, cdf_max);

    float* uniqueIntensity = (float*)malloc(256 * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
        uniqueIntensity[i] = -1;
    }

    #pragma omp parallel for
    for (int x = 0; x < src.rows; x++) {
        for (int y = 0; y < src.cols; y++) {
            int index = int(src.at<uchar>(x, y));
            uniqueIntensity[index] = 1;
        }
    }

    float prob_max = cuda_findMax(prob);
    float prob_min = cuda_findMin(prob);

    float* pn_temp = (float*)malloc(256 * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
        pn_temp[i] = (prob[i] - prob_min) / (prob_max - prob_min);
    }

    float pn_temp_sum = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
        if (pn_temp[i] > 0) {
            pn_temp[i] = prob_max * pow(pn_temp[i], alpha);
        }
        if (pn_temp[i] < 0) {
            pn_temp[i] = prob_max * (-pow((-pn_temp[i]), alpha));
        }
        pn_temp_sum += pn_temp[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
        pn_temp[i] /= pn_temp_sum;
    }

    float* prob_normalized_wd = (float*)malloc(256 * sizeof(float));
    prob_normalized_wd[0] = pn_temp[0];
    for (int i = 1; i < 256; i++) {
        prob_normalized_wd[i] = prob_normalized_wd[i-1] + pn_temp[i];
    }

    float* inverse_cdf = (float*)malloc(256 * sizeof(float));
    if (truncated_cdf) {
        #pragma omp parallel for
        for (int i = 0; i < 256; i++) {
            inverse_cdf[i] = (0.5 > 1 - prob_normalized_wd[i])? 0.5: 1 - prob_normalized_wd[i];
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < 256; i++) {
            inverse_cdf[i] = 1 - prob_normalized_wd[i];
        }
    }

    cv::Mat dst;
    cuda_calc(src, dst, uniqueIntensity, inverse_cdf);

    free(hist);
    free(prob);
    free(cdf);
    free(norm_cdf);
    free(uniqueIntensity);
    free(pn_temp);
    free(prob_normalized_wd);
    free(inverse_cdf);

    return dst;
}
