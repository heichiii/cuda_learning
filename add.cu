#include <iostream>
#define N 10

__global__ void addd(int *a,int *b, int* c)
{
    int tid = blockIdx.x;
    
    if(tid<N)
    {
        c[tid]=a[tid]+b[tid];
    }
    

}

int main()
{
    int a[N]={1,2,3,4,5,6,7,8,9,0};
    int b[N]={0,9,8,7,6,5,4,3,2,1};
    int c[N]={0};
    int *dev_a;
    int *dev_b;
    int* dev_c;
    std::cout<<"malloc start"<<std::endl;
    cudaMalloc((void **)&dev_c,sizeof(int)*N);
    cudaMalloc((void **)&dev_b,sizeof(int)*N);
    cudaMalloc((void **)&dev_a,sizeof(int)*N);
    std::cout<<"malloc finished"<<std::endl;
    cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice);
    std::cout<<"copy finished"<<std::endl;
    addd<<<N,1>>>(dev_a,dev_b,dev_c);
    std::cout<<"add finished"<<std::endl;
    cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
    {
        std::cout<<c[i]<<std::endl;
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;

    
}