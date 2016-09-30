
#include "cuda_runtime.h"	// cuda����ʱAPI
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    //int i = threadIdx.x;	// �����̲߳��еĴ���
	int i = blockIdx.x;	// ���ǿ鲢�еĴ���
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	// Add vectors in parallel
	// ���Ľ�����: ������豸����ʶ���������޸�
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
// �ص�����������
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;	// GPU�豸������ָ��
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;	// ״ָ̬ʾ

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);	// ѡ������ƽ̨
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	// ����GPU�豸���ڴ�
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
	// �������ݵ�GPU
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
	// ���к˺���
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);   //ԭ����.

	//addKernel << <1, size >> >(dev_c, dev_a, dev_b);  // �����:�̲߳��� 
	//addKernel << <size, 1 >> >(dev_c, dev_a, dev_b);	// ������:�鲢��
	
	cudaStream_t stream[5];	// ���߽�:������
	for (int i = 0; i < 5; i++)
	{
		cudaStreamCreate(&stream[i]);	// ������
	}

	for (int i = 0; i < 5; i++)
	{
		addKernel << <1, 1, 0, stream[i] >> >(dev_c + i, dev_a + i, dev_b + i);	// ִ����
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
	// �����ṹ������.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	// ���߽�:������
	for (int i = 0; i < 5; i++)
	{
		cudaStreamDestroy(stream[i]);	// ������
	}

    cudaFree(dev_c);	// �ͷ�GPU�豸���ڴ�
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
