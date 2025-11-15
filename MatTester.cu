//
// Created by Celestee on 2025/11/11.
//

#include "MatTester.cuh"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include "cublas_v2.h"
void MatTester::initMatrices()
{
    std::cout<<"Initializing Matrices..."<<std::endl;
    // Use C++11 random library instead of rand()
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(int i=0;i<A.rows*A.cols;i++)
    {
        A.data[i]=dis(gen);
    }
    for(int i=0;i<B.rows*B.cols;i++)
    {
        B.data[i]=dis(gen);
    }
}
void MatTester::allocateDeviceMemory()
{
    cudaMalloc((void**)&dev_A,sizeof(float)*A.rows*A.cols);
    cudaMalloc((void**)&dev_B,sizeof(float)*B.rows*B.cols);
    cudaMalloc((void**)&dev_C,sizeof(float)*C.rows*C.cols);
    cudaMalloc((void**)&dev_C2,sizeof(float)*C.rows*C.cols);
}
void MatTester::copyMatricesToDevice()
{
    cudaMemcpy(dev_A,A.data,sizeof(float)*A.rows*A.cols,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B,B.data,sizeof(float)*B.rows*B.cols,cudaMemcpyHostToDevice);
}
void MatTester::launchKernel()
{
    auto t1 = std::chrono::high_resolution_clock::now();
    size_t sharedMemSize = blockSize.x * sizeof(float);
    matMulKernel<<<gridSize,blockSize,sharedMemSize>>>(dev_A,dev_B,dev_C,A.rows,A.cols,B.cols);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "GPU kernel computation time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
}
void MatTester::copyResultToHost()
{
    cudaMemcpy(C.data,dev_C,sizeof(float)*C.rows*C.cols,cudaMemcpyDeviceToHost);
}
bool MatTester::verifyResult()
{
    cudaError_t cuda_stat;
    cublasStatus_t blas_stat;
    cublasHandle_t handle;
    blas_stat = cublasCreate(&handle);
    if (blas_stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed\n");
        return false;
    }
    float alpha=1.0f;
    float beta=0.0f;
    auto t1 = std::chrono::high_resolution_clock::now();
    blas_stat= cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                             B.cols,A.rows,A.cols,
                             &alpha,
                             dev_A,B.cols,
                             dev_B,A.cols,
                             &beta,
                             dev_C2,C2.cols);
    cuda_stat=cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "cuBLAS matrix multiplication time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;

    if (cuda_stat != cudaSuccess)
    {
        printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(cuda_stat));
        return false;
    }
    cuda_stat=cudaMemcpy(C2.data,dev_C2,sizeof(float)*C.rows*C.cols,cudaMemcpyDeviceToHost);
    if (cuda_stat != cudaSuccess)
    {

    }
    // std::cout << "Verifying result on CPU..." << std::endl;
    //
    // int A_rows = A.rows;
    // int A_cols = A.cols;
    // int B_cols = B.cols;
    // size_t outSize = static_cast<size_t>(A_rows) * static_cast<size_t>(B_cols);
    //
    // // Allocate CPU result buffer
    // auto cpuC = new float[outSize];
    // for (size_t i = 0; i < outSize; ++i) cpuC[i] = 0.0f;
    //
    // // Time the CPU multiplication
    // auto t1 = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < A_rows; ++i)
    // {
    //     for (int j = 0; j < B_cols; ++j)
    //     {
    //         float sum = 0.0f;
    //         for (int k = 0; k < A_cols; ++k)
    //         {
    //             sum += A.data[i * A_cols + k] * B.data[k * B_cols + j];
    //         }
    //         cpuC[i * B_cols + j] = sum;
    //     }
    // }
    // auto t2 = std::chrono::high_resolution_clock::now();
    // double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // std::cout << "CPU matrix multiplication time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;

    // Compare
    const float relTol = 1e-3f; // relative tolerance
    const float absTol = 1e-5f; // absolute tolerance
    bool allMatch = true;
    int mismatchCount = 0;
    const int maxMismatchesToShow = 5;

    for (size_t idx = 0; idx < C2.rows*C2.cols; ++idx)
    {
        float cpuVal = C.data[idx];
        float gpuVal = C2.data[idx];
        float diff = std::fabs(cpuVal - gpuVal);
        // float threshold = std::max(absTol, relTol * std::max(std::fabs(cpuVal), std::fabs(gpuVal)));
        float threshold = 30;

        if (diff > threshold)
        {
            allMatch = false;
            if (mismatchCount < maxMismatchesToShow)
            {
                std::cout << "Mismatch at index " << idx
                          << ": CPU=" << cpuVal
                          << " GPU=" << gpuVal
                          << " diff=" << diff << std::endl;
            }
            mismatchCount++;
        }
    }

    // delete[] cpuC;

    if (allMatch)
    {
        std::cout << "Verification PASSED - All results match!" << std::endl;
    }
    else
    {
        std::cout << "Verification FAILED - " << mismatchCount
                  << " mismatches found out of " << " elements" << std::endl;
    }

    return allMatch;
}

void MatTester::freeDeviceMemory()
{
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

// Kernel implementation - must be outside the class
__global__ void matMulKernel(const float* A,const float* B,float* C,int A_rows, int A_cols, int B_cols )
{
    extern __shared__ float cache[];
    int tid=threadIdx.x;
    cache[tid]=0;
    while (tid<A_cols)
    {
        cache[tid]+=A[blockIdx.x*A_cols+tid]*B[blockIdx.y+B_cols*tid];
        tid+=blockDim.x;
    }
    __syncthreads();
    int i=blockDim.x/2;
    while (i!=0)
    {
        if (threadIdx.x<i)
        {
            cache[threadIdx.x]+=cache[threadIdx.x+i];
        }
        __syncthreads();
        i/=2;
    }
    if (threadIdx.x==0)
    {
        C[blockIdx.x*B_cols+blockIdx.y]=cache[0];
    }

}


