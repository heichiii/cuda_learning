//
// Created by Celestee on 2025/11/11.
//

#ifndef CUDA_LEARNING_MATTESTER_CUH
#define CUDA_LEARNING_MATTESTER_CUH

#include "cuda_runtime.h"

struct  Matrix
{
    int rows,cols;
    float *data;
};

// Kernel declaration - must be outside the class
__global__ void matMulKernel(const float* A,const float* B,float* C, int A_rows, int A_cols, int B_cols );

class MatTester
{
public:

    MatTester(int A_rows,int A_cols,int B_cols):A{A_rows,A_cols},B{A_cols,B_cols},C{A_rows,B_cols},C2{A_rows,B_cols},N(A_rows*B_cols*A_cols)
    {
        A.data=new float[A_rows*A_cols];
        B.data=new float[A_cols*B_cols];
        C.data=new float[A_rows*B_cols];
        C2.data=new float[A_rows*B_cols];
        gridSize=dim3(A_rows,B_cols);//假设不超出限制
        blockSize=dim3(std::min(A_cols,1024));
    }
    ~MatTester()
    {
        delete[] A.data;
        delete[] B.data;
        delete[] C.data;
        delete[] C2.data;
    }
    void initMatrices();
    void allocateDeviceMemory();
    void copyMatricesToDevice();
    void launchKernel();
    void copyResultToHost();
    bool verifyResult();
    void freeDeviceMemory();
    inline size_t transIndex(int row,int col,int nrows)
    {
        return row+col*nrows;
    }

private:
    Matrix A,B,C,C2;
    size_t N; //number of "threads"
    dim3 blockSize;
    dim3 gridSize;
    float* dev_A{nullptr};
    float* dev_B{nullptr};
    float* dev_C{nullptr};
    float* dev_C2{nullptr};
};


#endif //CUDA_LEARNING_MATTESTER_CUH