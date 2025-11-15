#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", dev, prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    return 0;
}