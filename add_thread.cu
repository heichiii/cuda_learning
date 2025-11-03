#include <iostream>

#define N (33*1024)

__global__ void add(int *a,int* b,int *c)
{
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    while (tid<N)
    {
        c[tid]=a[tid]+b[tid];
        tid+=blockDim.x*gridDim.x;
    }
    
}

int main()
{
    int a[N];
    int b[N];
    int c[N];
    int *dev_a;
    int *dev_b;
    int* dev_c;
    std::cout<<"malloc start"<<std::endl;
    cudaMalloc((void **)&dev_c,sizeof(int)*N);
    cudaMalloc((void **)&dev_b,sizeof(int)*N);
    cudaMalloc((void **)&dev_a,sizeof(int)*N);
    std::cout<<"malloc finished"<<std::endl;

    for(int i=0;i<N;i++)
    {
        a[i]=i;
        b[i]=i*i;
    }


    cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice);
    std::cout<<"copy finished"<<std::endl;
    add<<<128,128>>>(dev_a,dev_b,dev_c);
    std::cout<<"add finished"<<std::endl;
    cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
    {
        // std::cout<<c[i]<<std::endl;
        if((a[i]+b[i])!=c[i])
        {
            std::cout<<"Error:"<<a[i]<<" +"<<b[i]<<" != "<<c[i]<<std::endl;
        }
    }
    std::cout<<"all finished"<<std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;

    
}