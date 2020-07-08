#include <cuda_runtime.h>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_atomic_functions.h>
#include <helper_math.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include "cuda_simulation.cuh"
#include "sph_kernel.cuh"

namespace cg = cooperative_groups;

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.world_origin.x) / params.cell_size.x);
	gridPos.y = floor((p.y - params.world_origin.y) / params.cell_size.y);
	gridPos.z = floor((p.z - params.world_origin.z) / params.cell_size.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.grid_size.y - 1);
	gridPos.z = gridPos.z & (params.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, params.grid_size.y), params.grid_size.x) + __umul24(gridPos.y, params.grid_size.x) + gridPos.x;
}

// collide two spheres using DEM method
__device__
float3 collideSpheres(
	float3 posA, float3 posB,
	float3 velA, float3 velB,
	float radiusA, float radiusB,
	float attraction)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	//printf("dist: %f\ncollideDist: %f", dist, collideDist);

	if (dist < collideDist)
	{
		float3 norm = relPos / (dist+0.00001f);

		// relative velocity
		float3 relVel = velB - velA;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -params.spring * (collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping * relVel;
		// tangential shear force
		force += params.shear * tanVel;
		// attraction
		force += attraction * relPos;

		//printf("%f %f %f\n", force.x, force.y, force.z);
	}

	return force;
}

__device__
float3 collideCell(
	int3    gridPos,
	uint    index,
	float3  pos,
	float3  vel,
	float3* oldPos,
	float3* oldVel,
	uint* cellStart,
	uint* cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = oldPos[j];
				float3 vel2 = oldVel[j];

				// collide two spheres
				force += collideSpheres(
					pos, pos2,
					vel, vel2,
					params.particle_radius, params.particle_radius,
					params.attraction);
			}
		}
	}

	return force;
}

__global__ void calcHashD(
	uint* grid_particle_hash,  // output
	uint* grid_particle_index, // output
	float3* pos,               // input: positions
	uint    num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles) return;

	volatile float3 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	grid_particle_hash[index] = hash;
	grid_particle_index[index] = index;
}

/*
 * Reorder data to find cell start and end (for neighbor searching)
 */
__global__
void reorderDataAndFindCellStartD(
	uint* cellStart,        // output: cell start index
	uint* cellEnd,          // output: cell end index
	float3* sortedPos,        // output: sorted positions
	float3* sortedVel,        // output: sorted velocities
	uint* gridParticleHash, // input: sorted grid hashes
	uint* gridParticleIndex,// input: sorted particle indices
	float3* oldPos,           // input: sorted position array
	float3* oldVel,           // input: sorted velocity array
	uint    numParticles)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	cg::sync(cta);

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		float3 pos = oldPos[sortedIndex];
		float3 vel = oldVel[sortedIndex];

		sortedPos[index] = pos;
		sortedVel[index] = vel;
	}
}

void compute_grid_size(uint n, uint block_size, uint& num_blocks, uint& num_threads)
{
	num_threads = min(block_size, n);
	num_blocks = (n % num_threads != 0) ? (n / num_threads + 1) : (n / num_threads);
}

void calculate_hash(
	uint* grid_particle_hash,
	uint* grid_particle_index,
	float3* pos,
	uint    num_particles)
{
	uint num_blocks, num_threads;
	compute_grid_size(num_particles, MAX_THREAD_NUM, num_blocks, num_threads);
	calcHashD << < num_blocks, num_threads >> > (
		grid_particle_hash,
		grid_particle_index,
		pos,
		num_particles);
}

void reorderDataAndFindCellStart(
	uint* cellStart,
	uint* cellEnd,
	float3* sortedPos,
	float3* sortedVel,
	uint* gridParticleHash,
	uint* gridParticleIndex,
	float3* oldPos,
	float3* oldVel,
	uint	numParticles,
	uint	numCells)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, 256, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint) * (numThreads + 1);
	reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> > (
		cellStart,
		cellEnd,
		sortedPos,
		sortedVel,
		gridParticleHash,
		gridParticleIndex,
		oldPos,
		oldVel,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}

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

__global__
void integrate_d(
	float3* pos, float3* vel, 
	float deltaTime, 
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 t_pos = pos[index];
	float3 t_vel = vel[index];
	
	if (index >= numParticles)
		return;
	/*
	if (index == 0)
		printf("particles[0]: %5f, %5f, %5f %5f\n", pos[index].x, pos[index].y, pos[index].z, params.gravity.y);
	*/
	t_vel = t_vel + params.gravity * deltaTime;
	//t_vel = params.damping * t_vel;
	t_pos = t_pos + t_vel * deltaTime;

	if (t_pos.x > 10.0f - params.particle_radius)
	{
		t_pos.x = 10.0f - params.particle_radius;
		t_vel.x *= params.boundary_damping;
	}

	if (t_pos.x < -10.0f + params.particle_radius)
	{
		t_pos.x = -10.0f + params.particle_radius;
		t_vel.x *= params.boundary_damping;
	}

	
	if (t_pos.z > 1.0f - params.particle_radius)
	{
		t_pos.z = 1.0f - params.particle_radius;
		t_vel.z *= params.boundary_damping;
	}

	if (t_pos.z < -15.0f + params.particle_radius)
	{
		t_pos.z = -15.0f + params.particle_radius;
		t_vel.z *= params.boundary_damping;
	}

	if (t_pos.y < 0.f + params.particle_radius)
	{
		t_pos.y = 0.f + params.particle_radius;
		t_vel.y *= params.boundary_damping;
	}

	pos[index] = t_pos;
	vel[index] = t_vel;
	/*
	if (index == 0)
		printf("After particles[0]: %f, %f, %f\n", pos[index].x, pos[index].y, pos[index].z);
		*/
}

__global__ 
void integrate_pbd_d(
	float3* pos, float3* vel, float3* force, float* massInv,
	float3* predict_pos, float3* new_pos,
	float dt,
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	float3 t_vel = vel[index] + dt * make_float3(0, -9.81f, 0);
	float3 t_pos = pos[index] + dt * t_vel;
	//vel[index] = t_vel;
	//predict_pos[index] = t_pos;
	//new_pos[index] = t_pos;

	float3 v_r = t_vel;
	float3 n;

	if (t_pos.x > 10.0f)
	{
		t_pos.x = 10.0f;
		//t_vel.x *= params.boundary_damping;
		n = make_float3(-1.f, 0.f, 0.f);
		v_r = t_vel - 2.f * dot(t_vel, n) * n;
	}

	if (t_pos.x < -10.0f)
	{
		t_pos.x = -10.0f;
		//t_vel.x *= params.boundary_damping;
		n = make_float3(1.f, 0.f, 0.f);
		v_r = t_vel - 2.f * dot(t_vel, n) * n;
	}


	if (t_pos.z > 1.0f)
	{
		t_pos.z = 1.0f;
		//t_vel.z *= params.boundary_damping;
		n = make_float3(0.f, 0.f, -1.f);
		v_r = t_vel - 2.f * dot(t_vel, n) * n;
	}

	if (t_pos.z < -15.0f)
	{
		t_pos.z = -15.0f;
		//t_vel.z *= params.boundary_damping;
		n = make_float3(0.f, 0.f, 1.f);
		v_r = t_vel - 2.f * dot(t_vel, n) * n;
	}

	if (t_pos.y < 0.f)
	{
		t_pos.y = 0.f;
		//t_vel.y *= params.boundary_damping;
		n = make_float3(0.f, 1.f, 0.f);
		v_r = t_vel - 2.f * dot(t_vel, n) * n;
	}

	vel[index] = params.boundary_damping * v_r;
	predict_pos[index] = pos[index] + dt * v_r;
	new_pos[index] = predict_pos[index];


}

// collide a particle against all other particles in a given cell
/* Collision device code */
__global__
void collideD(
	float3* newVel,               // output: new velocity
	float3* oldPos,               // input: sorted positions
	float3* oldVel,               // input: sorted velocities
	uint* gridParticleIndex,      // input: sorted particle indices
	uint* cellStart,
	uint* cellEnd,
	uint  numParticles,
	float dt)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = oldPos[index];
	float3 vel = oldVel[index];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);

	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				force += collideCell(neighbor_pos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
			}
		}
	}

	// write new velocity back to original unsorted location
	uint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = vel + force * dt; // + force/mass * dt ?
}

__device__
float pbf_density(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3* sorted_pos,
	float*	mass,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float density = 0.0f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];
				
				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);
				
				float rho = mass[original_index] * Poly6_W_CUDA(dist, params.particle_radius);

				density += rho;
			}
		}
	}

	return density;
}

__device__
float pbf_lambda(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float*	rest_density,
	float3* sorted_pos,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float gradientC_sum = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];

				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradientC_j = (1.f / (*rest_density)) *
					Poly6_W_Gradient_CUDA(vec, dist, params.particle_radius);

				float dot_val = dot(gradientC_j, gradientC_j);
				gradientC_sum += dot_val;
			}
		}
	}
	return gradientC_sum;
}

__device__
float3 pbf_correction(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	lambda_i,
	float*	rest_density,
	float3* sorted_pos,
	float*	lambda,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];

				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradient = Poly6_W_Gradient_CUDA(vec, dist, params.particle_radius);
				
				float scorr = -0.1f;
				float x = Poly6_W_CUDA(dist, params.particle_radius) / 
					Poly6_W_CUDA(0.3f * params.particle_radius, params.particle_radius);
				x = pow(x, 4);
				scorr = scorr * x * dt;
				
				float3 res = (1.f / (*rest_density)) *
					(lambda_i + lambda[original_index] + scorr) *
					gradient;
				
				correction += res;
			}
		}
	}
	return correction;
}

__global__
void compute_rest_density_d(
	float*	rest_density,			// output: rest density
	float3* sorted_pos,				// input: sorted mass
	float*	mass,					// input: mass
	uint*	gridParticleIndex,		// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];

	// initial density
	float rho = mass[originalIndex] * Poly6_W_CUDA(0, params.particle_radius);

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				rho += pbf_density(
					neighbor_pos, index,
					pos, sorted_pos, mass,
					cellStart, cellEnd, gridParticleIndex
				);
			}
		}
	}

	//printf("rho[%u]: %f", index, rho);

	atomicAdd(rest_density, rho/(float)numParticles);

}

__global__
void compute_density_d(
	float*	density,					// output: computed density
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	float*	mass,						// input: mass
	float*	C,							// input: contraint
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];
	
	// initial density
	float rho = mass[originalIndex] * Poly6_W_CUDA(0, params.particle_radius);

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				rho += pbf_density(
					neighbor_pos, index, 
					pos, sorted_pos, mass, 
					cellStart, cellEnd, gridParticleIndex
				);
			}
		}
	}

	// Update date density and constraint value
	density[originalIndex] = rho;
	C[originalIndex] = (rho / (*rest_density)) - 1.f;

	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);

}

__global__
void compute_lambdas_d(
	float*	lambda,						// output: computed density
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	float*	C,							// input: contraint
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];

	// initial density
	lambda[originalIndex] = -C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	const float epislon = 1.f;
	float3 gradientC_i = -(1.f / (*rest_density)) *
		Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.particle_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);
	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda(
					neighbor_pos, index,
					pos, rest_density,
					sorted_pos,
					cellStart, cellEnd, gridParticleIndex
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	lambda[originalIndex] /= gradientC_sum;// +epislon;

	//lambda[originalIndex] = lambda_res;
}

__global__
void compute_position_correction(
	float*	lambda,						// output: computed density
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	float3* new_pos,					// output: new_pos
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles,
	float	dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];

	// initial density
	float lambda_i = lambda[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 correction = make_float3(0, 0, 0);


	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				correction += pbf_correction(
					neighbor_pos, index,
					pos, lambda_i, rest_density,
					sorted_pos, lambda,
					cellStart, cellEnd, gridParticleIndex,
					dt
				);
			}
		}
	}

	//compute new position
	new_pos[originalIndex] += correction;
}

__global__
void apply_correction(
	float3* new_pos,
	float3* predict_pos,
	uint numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	
	predict_pos[index] = new_pos[index];	
}


__global__
void finalize_correction(
	float3* pos,
	float3* new_pos,
	float3* predict_pos,
	float3* velocity,
	uint numParticles,
	float dt
) 
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	//float3 res = new_pos[index];
	//float3 vel = (res - pos[index]) / dt;

	float3 t_pos = new_pos[index];
	float3 t_vel = (t_pos - pos[index]) / dt;
	
	

	velocity[index] = t_vel;
	predict_pos[index] = t_pos;
	pos[index] = t_pos;


}

void allocateArray(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void setParams(SimParams* param_in)
{
	checkCudaErrors(cudaMemcpyToSymbol(params, param_in, sizeof(SimParams)));
}

void integrate(float3* pos, float3* vel, float deltaTime, uint numParticles)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	integrate_d << <numBlocks, numThreads >> > (
		pos, vel,
		deltaTime,
		numParticles
		);
}

/* Integration for Position based Dynamics */
void integratePBD(
	float3* pos, float3* vel,  
	float3* force, float* massInv,
	float3* predict_pos, float3* new_pos,
	float deltaTime,
	uint numParticles
)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	integrate_pbd_d << <numBlocks, numThreads >> > (
		pos, vel, force, massInv,
		predict_pos, new_pos,
		deltaTime,
		numParticles
		);
}

void sort_particles(uint* dGridParticleHash, uint* dGridParticleIndex, uint numParticles)
{
	//printf("uint size: %u\n", sizeof(uint));
	thrust::sort_by_key(
		thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex)
	);
}

void solve_dem_collision(
	float3* newVel,
	float3* sortedPos,
	float3* sortedVel,
	uint*	gridParticleIndex,
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles,
	uint	numCells,
	float	dt)
{

	// thread per particle
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	// execute the kernel
	collideD << < numBlocks, numThreads >> > (
		newVel,
		sortedPos,
		sortedVel,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles,
		dt
	);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

}

void compute_rest_density(
	float* rest_density,			// output: rest density
	float3* sorted_pos,				// input: sorted mass
	float* mass,					// input: mass
	uint* gridParticleIndex,		// input: sorted particle indices
	uint* cellStart,
	uint* cellEnd,
	uint  numParticles
)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);
	compute_rest_density_d << <numBlocks, numThreads >> > (
		rest_density,							// output
		sorted_pos, mass,						// input particle information
		gridParticleIndex, cellStart, cellEnd,	// input cell info
		numParticles		
		);
}

void solve_sph_fluid(
	float3* pos,
	float3* new_pos,
	float3* predict_pos,
	float3* vel,
	float3* sorted_pos,
	float3* sorted_vel,
	float*	mass,
	float*	density,
	float*	rest_density,
	float*	C,
	float*	lambda,
	uint*	gridParticleIndex,
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles,
	uint	numCells,
	float	dt)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	// CUDA SPH Kernel
	// compute density
	compute_density_d <<<numBlocks, numThreads>>>(
		density, rest_density,
		sorted_pos,
		mass, C,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles
	);

	// compute lambda
	compute_lambdas_d <<<numBlocks, numThreads >>>(
		lambda,
		rest_density,
		sorted_pos,
		C,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles
	);

	// compute new position
	compute_position_correction << <numBlocks, numThreads >> > (
		lambda,
		rest_density,
		sorted_pos,
		new_pos,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles,
		dt
	);
	
	// correct this iteration
	//apply_correction << <numBlocks, numThreads >> > (new_pos, predict_pos, numParticles);

	// final correction
	finalize_correction << <numBlocks, numThreads >> > (
		pos, new_pos, predict_pos, vel, 
		numParticles, 
		dt
	);
	

}

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions)
{
	test_offset << <block_num, thread_num >> > (positions);
}
