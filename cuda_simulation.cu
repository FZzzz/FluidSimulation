#include <cuda_runtime.h>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include "cuda_simulation.cuh"

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

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

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

void sort_particles(uint* dGridParticleHash, uint* dGridParticleIndex, uint numParticles)
{
	printf("uint size: %u\n", sizeof(uint));
	thrust::sort_by_key(
		thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex)
	);
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

	t_vel = t_vel + params.gravity * deltaTime;
	t_vel = params.damping * t_vel;
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
	uint  numParticles)
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
	newVel[originalIndex] = vel + force; // + force/mass * dt ?
}

void allocateArray(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}


void setParams(SimParams* param_in)
{
	cudaMemcpyToSymbol(&params, param_in, sizeof(SimParams));
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

void collide(
	float3* newVel,
	float3* sortedPos,
	float3* sortedVel,
	uint*	gridParticleIndex,
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles,
	uint	numCells)
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
		numParticles
	);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

}

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions)
{
	test_offset << <block_num, thread_num >> > (positions);
}
