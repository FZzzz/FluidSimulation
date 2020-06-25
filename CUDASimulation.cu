#include "CUDASimulation.cuh"
#include <stdio.h>

__global__ void test_offset(float3* positions)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/*
	if (i == 0)
		printf("particles[0]: %f, %f, %f\n", positions[i].x , positions[i].y, positions[i].z);
	*/
	positions[i].x = positions[i].x + 0.001f;
	positions[i].y = positions[i].y + 0.001f;
	positions[i].z = positions[i].z + 0.001f;

}


void cuda_simulation()
{
}

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions)
{
	test_offset << <block_num, thread_num >> > (positions);
}
