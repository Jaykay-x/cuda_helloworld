
#include "cuda_runtime.h"	// cuda运行时API
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    //int i = threadIdx.x;	// 这是线程并行的代码
	int i = blockIdx.x;	// 这是块并行的代码
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	// Add vectors in parallel
	// 第四节里面: 加深对设备的认识里所做的修改
	cudaError_t cudaStatus;
	int num = 0;
	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceCount(&num);

	for (int i = 0; i < num; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		printf("This DisplayCard is: %s\n\nClockRate is: %d \n\nCuda Version:%d.%d\n\n", prop.name, prop.clockRate, prop.major, prop.minor);
	}
	cudaStatus = addWithCuda(c, a, b, arraySize);

    // Add vectors in parallel.
    /*cudaError_t*/ cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
// 重点理解这个函数
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;	// GPU设备端数据指针
    int *dev_b = 0;
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

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
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

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	// 运行核函数
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);   //原来的.

	//addKernel << <1, size >> >(dev_c, dev_a, dev_b);  // 第五节:线程并行 
	//addKernel << <size, 1 >> >(dev_c, dev_a, dev_b);	// 第六节:块并行
	
	cudaStream_t stream[5];	// 第七节:流并行
	for (int i = 0; i < 5; i++)
	{
		cudaStreamCreate(&stream[i]);	// 创建流
	}

	for (int i = 0; i < 5; i++)
	{
		addKernel << <1, 1, 0, stream[i] >> >(dev_c + i, dev_a + i, dev_b + i);	// 执行流
	}

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
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
	// 第七节:流并行
	for (int i = 0; i < 5; i++)
	{
		cudaStreamDestroy(stream[i]);	// 销毁流
	}

    cudaFree(dev_c);	// 释放GPU设备端内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
