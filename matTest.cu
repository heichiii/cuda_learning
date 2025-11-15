//
// Created by Celestee on 2025/11/11.
//
#include <iostream>
#include "MatTester.cuh"

int main() {
    MatTester tester1(1024,1024,1024);
    tester1.initMatrices();
    tester1.allocateDeviceMemory();
    tester1.copyMatricesToDevice();
    tester1.launchKernel();
    tester1.copyResultToHost();
    if(tester1.verifyResult())
    {
        std::cout<<"Test passed!"<<std::endl;
    }
    else
    {
        std::cout<<"Test failed!"<<std::endl;
    }
    tester1.freeDeviceMemory();
    return 0;
}