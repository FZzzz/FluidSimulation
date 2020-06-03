#ifndef _SPH_KERNEL_H_
#define _SPH_KERNEL_H_

#include "common.h"

class SPHKernel
{
public:
	static float Poly6_W(float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			float h2 = effective_radius * effective_radius;
			float h9 = glm::pow(effective_radius, 9);
			float d2 = distance * distance;
			float q = h2 - d2;
			float q3 = q * q * q;
			
			return (315.f / (64.f * M_PI * h9)) * q3;
		}
		else
		{
			return 0.0f;
		}
	}

	static glm::vec3 Poly6_W_Gradient(glm::vec3 diff, float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			float h2 = effective_radius * effective_radius;
			float h9 = glm::pow(effective_radius, 9);
			float d2 = distance * distance;
			float q = h2 - d2;
			float q2 = q * q;

			return (-945.f/ (32.f * M_PI * h9)) * q2 * diff;
		}
		else
		{
			return glm::vec3(0);
		}
	}

	static float Spiky_W(float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			float h6 = glm::pow(effective_radius, 6);
			float q = effective_radius - distance;
			float q3 = q * q * q;

			return (15.f / (M_PI * h6)) * q3;
		}
		else
		{
			return 0.0f;
		}
	}


	static glm::vec3 Spiky_W_Gradient(glm::vec3 diff, float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			float h6 = glm::pow(effective_radius, 6);
			float q = effective_radius - distance;
			float q2 = q * q;

			return (-45.f / (M_PI * h6)) * (q2/distance) * diff;
		}
		else
		{
			return glm::vec3(0);
		}
	}


};

#endif
