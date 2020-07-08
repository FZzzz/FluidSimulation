#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <vector_functions.h>

__device__ float Poly6_W_CUDA(float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const double h = (double)(effective_radius);
		const double d = (double)(distance);

		double h2 = h * h;
		double h9 = pow(h, 9);
		double d2 = d * d;
		double q = h2 - d2;
		double q3 = q * q * q;

		double result = (315.0 / (64.0 * CUDART_PI * h9)) * q3;

		return (float)(result);
	}
	else
	{
		return 0.0f;
	}
}

__device__ float3 Poly6_W_Gradient_CUDA(float3 diff, float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const double h = (double)(effective_radius);
		const double d = (double)(distance);

		double h2 = h * h;
		double h9 = pow(h, 9);
		double d2 = d * d;
		double  q = h2 - d2;
		double q2 = q * q;

		double scalar = (-945.0 / (32.0 * CUDART_PI * h9));
		scalar = scalar * q2;
		float3 result = make_float3(scalar * diff.x, scalar * diff.y, scalar * diff.z);

		return result;
	}
	else
	{
		return make_float3(0,0,0);
	}
}


__device__ float Spiky_W_CUDA(float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const double h = (double)(effective_radius);
		const double d = (double)(distance);

		double h6 = pow(h, 6);
		double q = h - d;
		double q3 = q * q * q;

		float result = (float)((15.0 / (CUDART_PI * h6)) * q3);

		return result;
	}
	else
	{
		return 0.0f;
	}
}


__device__ float3 Spiky_W_Gradient_CUDA(float3 diff, float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const double h = (double)(effective_radius);
		const double d = (double)(distance);
		double h6 = pow(h, 6);
		double q = h - d;
		double q2 = q * q;

		double scalar = (-45.0 / (CUDART_PI*h6)) * (q2 / distance);
		float3 result = make_float3(scalar*diff.x, scalar*diff.y, scalar*diff.z);

		return result;
	}
	else
	{
		return make_float3(0,0,0);
	}
}