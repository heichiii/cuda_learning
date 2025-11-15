//
// Created by Celestee on 2025/11/8.
//
#include <iostream>
// #include <cuda_runtime.h>

// 定义矩阵维度
const int A_rows = 32;
const int A_cols = 32;
const int B_rows = A_cols;
const int B_cols = 32;


// CUDA核函数：矩阵乘法
__global__ void matrixMultiplyDot(float *a, float *b, float *c) {
    __shared__ float cache[A_cols];

    cache[threadIdx.x]=a[blockIdx.x*A_cols+threadIdx.x]*b[blockIdx.y+B_cols*threadIdx.x];
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
        c[blockIdx.x*B_cols+blockIdx.y]=cache[0];
    }
}

// 初始化矩阵
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = rand() % 10;  // 0-9的随机数
        }
    }
}

// 验证结果正确性
bool verifyResult(float *a, float *b, float *c) {
    float *verify_c = new float[A_rows * B_cols];

    // CPU计算矩阵乘法
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            int sum = 0;
            for (int k = 0; k < A_cols; k++) {
                sum += a[i * A_cols + k] * b[k * B_cols + j];
            }
            verify_c[i * B_cols + j] = sum;
        }
    }

    // 比较结果
    bool correct = true;
    for (int i = 0; i < A_rows * B_cols; i++) {
        if (c[i] != verify_c[i]) {
            correct = false;
            break;
        }
    }

    delete[] verify_c;
    return correct;
}

int main() {
    // 分配主机内存
    float *h_a = new float[A_rows * A_cols];
    float *h_b = new float[B_rows * B_cols];
    float *h_c = new float[A_rows * B_cols];

    // 初始化输入矩阵
    initializeMatrix(h_a, A_rows, A_cols);
    initializeMatrix(h_b, B_rows, B_cols);



    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(float)*A_rows*A_cols);
    cudaMalloc(&d_b, sizeof(float)*B_rows*B_cols);
    cudaMalloc(&d_c, sizeof(float)*A_rows*B_cols);

    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, sizeof(float)*A_rows*A_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*B_rows*B_cols, cudaMemcpyHostToDevice);

    // 定义线程块和网格维度
    dim3 blockSize(A_cols);  // 每个块A_cols个线程
    dim3 gridSize(A_rows,B_cols); //C的每个元素是一个线程块

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动核函数
    cudaEventRecord(start);

    matrixMultiplyDot<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    // 等待核函数执行完成
    cudaDeviceSynchronize();

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, sizeof(float)*A_rows*B_cols, cudaMemcpyDeviceToHost);

    // std::cout << "结果矩阵 C:" << std::endl;
    // printMatrix(h_c, width);
    //
    // // 验证结果
    if (verifyResult(h_a, h_b, h_c)) {
        std::cout << "result correct!" << std::endl;
    } else {
        std::cout << "result error!" << std::endl;
    }

    std::cout << "execute time:" << milliseconds << " ms" << std::endl;

    // 清理资源
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}