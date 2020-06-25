#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <vector_functions.h>

void cuda_prediction();

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions);
