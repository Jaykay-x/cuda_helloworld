#include "cuda_runtime.h"	// cuda运行时API
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, size_t size);

__global__ void addKernel( int *c, const int *a )
{
    int i = threadIdx.x;	// 这是线程并行的代码
	
	extern __shared__ int smem[];
	smem[i]=a[i];
	__syncthreads();

	if (i == 0)	// 0号线程做平方和
	{
		c[0] = 0;
		for (int d = 0; d < 5; d++)
		{
			c[0] += smem[d] * smem[d];
		}
	}

	if (i == 1)	// 1号线程做累加
	{
		c[1] = 0;
		for (int d = 0; d < 5; d++)
		{
			c[1] += smem[d];
		}
	}
	

	if (i == 2)	// 2号线程做累乘
	{
		c[2] = 1;
		for (int d = 0; d < 5; d++)
		{
			c[2] *= smem[d];
		}
	}
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    int c[arraySize] = { 0 };

	cudaError_t cudaStatus = addWithCuda(c, a, arraySize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	
	printf("\t1+2+3+4+5 = %d\n\n\t1^2+2^2+3^2+4^2+5^2 = %d\n\n\t1*2*3*4*5 = %d\n\n\n\n\n\n\n",c[1],c[0],c[2]);
	
	cudaStatus = cudaThreadExit();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaThreadExit failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
// 重点理解这个函数
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
cudaError_t addWithCuda(int *c, const int *a, size_t size)
{
    int *dev_a = 0;	// GPU设备端数据指针
   
    int *dev_c = 0;
    cudaError_t cudaStatus;	// 状态指示

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);	// 选择运行平台
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	// 分配GPU设备端内存
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	
    // Copy input vectors from host memory to GPU buffers.
	// 拷贝数据到GPU
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	
	addKernel << <1, size, size*sizeof(int), 0 >> >(dev_c, dev_a);

	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

    // Copy output vector from GPU buffer to host memory.
	// 拷贝结构回主机.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	cudaFree(dev_c);	// 释放GPU设备端内存
    cudaFree(dev_a);
        
    return cudaStatus;
}
