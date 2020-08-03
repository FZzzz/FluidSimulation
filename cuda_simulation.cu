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
#include <chrono>
#include "imgui/imgui.h"

namespace cg = cooperative_groups;

// calculate position in uniform grid
inline __device__ int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.world_origin.x) / params.cell_size.x);
	gridPos.y = floor((p.y - params.world_origin.y) / params.cell_size.y);
	gridPos.z = floor((p.z - params.world_origin.z) / params.cell_size.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
inline __device__ uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.grid_size.y - 1);
	gridPos.z = gridPos.z & (params.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, params.grid_size.y), params.grid_size.x) + __umul24(gridPos.y, params.grid_size.x) + gridPos.x;
}

// collide two spheres using DEM method
inline __device__
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

inline __device__
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

inline __device__
float sph_boundary_volume(
	int3 grid_pos,
	uint index,
	float3 pos1,
	float* mass,
	CellData data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	uint start_index = data.cellStart[grid_hash];

	float rho = 0.f;

	if (start_index != 0xffffffff)
	{
		uint end_index = data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; ++j)
		{
			if (j != index)
			{
				uint original_index = data.grid_index[j];
				float3 pos2 = data.sorted_pos[j];
				float3 vec = pos1 - pos2;
				float dist = length(vec);
				rho += mass[original_index] * Poly6_W_CUDA(dist, params.effective_radius);
			}
		}
	}

	return rho;
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

__global__ 
void calcHash_boundary_D(
	CellData cell_data,
	float3* pos,               // input: positions
	uint    num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles) return;

	//printf("%u \n", index);

	volatile float3 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	cell_data.grid_hash[index] = hash;
	cell_data.grid_index[index] = index;
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

__global__
void reorderData_boundary_D(
	CellData cell_data,
	float3* oldPos,           // input: sorted position array
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
		hash = cell_data.grid_hash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = cell_data.grid_hash[index - 1];
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
			cell_data.cellStart[hash] = index;

			if (index > 0)
				cell_data.cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cell_data.cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos data
		uint sortedIndex = cell_data.grid_index[index];
		float3 pos = oldPos[sortedIndex];

		cell_data.sorted_pos[index] = pos;
	}
}
__global__
void compute_boundary_volume_d(
	CellData data, 
	float* mass, float* volume, 
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = data.sorted_pos[index];

	// initial volume
	float rho = mass[originalIndex] * Poly6_W_CUDA(0, params.effective_radius);

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
				rho += sph_boundary_volume(
					neighbor_pos, index,
					pos, mass,
					data
				);
			}
		}
	}

	// Update volume
	volume[originalIndex] = mass[originalIndex] / rho;

	//printf("rho = %f\n", rho);
	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);
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
	getLastCudaError("Kernel execution failed: calc_hash");
}

void calculate_hash_boundary(CellData cell_data, float3* pos, uint num_particles)
{
	uint num_blocks, num_threads;
	compute_grid_size(num_particles, MAX_THREAD_NUM, num_blocks, num_threads);

	calcHash_boundary_D << < num_blocks, num_threads >> > (
		cell_data,
		pos,
		num_particles);
	getLastCudaError("Kernel execution failed: calc_hash_boundary");
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
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

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

void reorderData_boundary(
	CellData cell_data, 
	float3* oldPos, 
	uint numParticles, 
	uint numCells)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cell_data.cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint) * (numThreads + 1);
	reorderData_boundary_D << < numBlocks, numThreads, smemSize >> > (
		cell_data,
		oldPos,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}

void compute_boundary_volume(CellData data, float* mass, float* volume, uint numParticles)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	compute_boundary_volume_d << <numBlocks, numThreads >> > (
		data,
		mass, volume,
		numParticles);

	getLastCudaError("Kernel execution failed: copmute_boundary_volume");
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
	
	float3 t_vel = vel[index] + dt * params.gravity;
	float3 t_pos = pos[index] + dt * t_vel;
	
	if (t_pos.x >= 1.0f)
	{
		t_pos.x = 1.f;
		t_vel.x =  -params.boundary_damping * abs(t_vel.x);
	}

	if (t_pos.x <= -1.0f)
	{
	 	t_pos.x = -1.f;
		t_vel.x = params.boundary_damping * abs(t_vel.x);
	}

	if (t_pos.z >= 1.0f)
	{
		t_pos.z = 1.f;
		t_vel.z = -params.boundary_damping * abs(t_vel.z);
	}

	if (t_pos.z <= -1.0f)
	{
		t_pos.z = -1.f;
		t_vel.z = params.boundary_damping * abs(t_vel.z);
	}
	
	if (t_pos.y <= 0.f)
	{
		t_pos.y = 0.f;
		t_vel.y = params.boundary_damping * abs(t_vel.y);
	}
	
	/* Velocity limitation
	if (length(t_vel) > 5.f)
	{
		t_vel = (5.f / length(t_vel)) * t_vel ;
	}
	*/
	
	predict_pos[index] = t_pos;// +dt * t_vel;
	vel[index] = t_vel;
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


inline __device__
float pbf_density_0(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3* sorted_pos,
	float*	mass,
	float*	rest_density,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex
) // type: 0->fluid fluid 1->boundary boundary 
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
				float rho = 0.f;

				rho = mass[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

				density += rho;
			}
		}
	}

	return density;
}

inline __device__
float pbf_density_1(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3* sorted_pos,
	float* mass,
	float* rest_density,
	uint* cell_start,
	uint* cell_end,
	uint* gridParticleIndex,
	float* b_volume = nullptr) // type: 0->fluid fluid 1->boundary boundary 
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
				float rho = 0.f;

				rho = (*rest_density) * b_volume[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

				density += rho;
			}
		}
	}

	return density;
}


inline __device__
float pbf_density_boundary(
	int3    grid_pos,
	float3  pos1,
	float* rest_density,
	float* volume,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cellStart[grid_hash];
	float density = 0.0f;

	// if cell of boundary cell data is not empty
	if (start_index != 0xffffffff)
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{	
			// no need to check collision (boundary cell data is not the same as fluid cell data)
			uint original_index = cell_data.grid_index[j];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float rho = (*rest_density) * volume[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

			density += rho;	
		}
	}

	// return contributions of boundary paritcles
	return density;
}

// boundary - fluid
inline __device__
float pbf_boundary_density(
	// boundary
	int3		grid_pos,	// searching grid pos
	float3		pos1,		// position of boundary particle
	// fluid
	float*		mass,
	float3*		sorted_pos,
	uint*		cell_start,
	uint*		cell_end,
	uint*		gridParticleIndex
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float density = 0.0f;

	// if cell of boundary cell data is not empty
	if (start_index != 0xffffffff)
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			// no need to check collision (boundary cell data is not the same as fluid cell data)
			uint original_index = gridParticleIndex[j];

			float3 pos2 = sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float rho = mass[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

			density += rho;
		}
	}

	// return contributions of boundary paritcles
	return density;
}

inline __device__
float pbf_lambda_0(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float*	rest_density,
	float*	mass,
	float3* sorted_pos,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex
)
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
				float particle_mass = mass[original_index];
				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradientC_j;

				gradientC_j = (1.f / (*rest_density)) *
					Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

				float dot_val = dot(gradientC_j, gradientC_j);
				gradientC_sum += dot_val;
			}
		}
	}
	return gradientC_sum;
}

inline __device__
float pbf_lambda_1(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float* rest_density,
	float* mass,
	float3* sorted_pos,
	uint* cell_start,
	uint* cell_end,
	uint* gridParticleIndex,
	float* b_volume = nullptr)
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
				float particle_mass = mass[original_index];
				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradientC_j;
				float vol = b_volume[original_index];

				gradientC_j = (1.f / (*rest_density)) *
						((*rest_density) * vol / particle_mass) *
						Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

				float dot_val = dot(gradientC_j, gradientC_j);
				gradientC_sum += dot_val;
			}
		}
	}
	return gradientC_sum;
}

// fluid - boundary
inline __device__
float pbf_lambda_boundary(
	int3		grid_pos,	// searching grid pos
	float3		pos1,		// position of fluid particle
	float*		rest_density,
	float		particle_mass,
	CellData	cell_data,	// cell data of boundary particle,
	float*		volume
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cellStart[grid_hash];
	float gradientC_sum = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			uint original_index = cell_data.grid_index[j];
			float vol = volume[original_index];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float3 gradientC_j = (1.f / (*rest_density)) * 
				((*rest_density) * vol / particle_mass) *  
				Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

			float dot_val = dot(gradientC_j, gradientC_j);
			gradientC_sum += dot_val;
		}
	}
	return gradientC_sum;
}

// Boundary - fluid 
inline __device__
float pbf_boundary_lambda(
	// boundary
	int3		grid_pos,	// searching grid pos
	float3		pos1,		// position of boundary particle
	float*		rest_density,
	float		particle_mass,
	float		volume,
	// fluid
	float3*		sorted_pos,  
	uint*		cell_start,
	uint*		cell_end,
	uint*		gridParticleIndex
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float gradientC_sum = 0.f;

	// search in fluid cell
	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float3 pos2 = sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float3 gradientC_j = (1.f / (*rest_density)) *
				Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

			float dot_val = dot(gradientC_j, gradientC_j);
			gradientC_sum += dot_val;
		}
	}
	return gradientC_sum;
}

inline __device__
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

				float3 gradient = Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);
				
				float scorr = -0.1f;
				float x = Poly6_W_CUDA(dist, params.effective_radius) / 
					Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
				x = pow(x, 4);
				scorr = scorr * x * dt * dt * dt;
				
				//printf("scorr: %f\n", scorr);

				float3 res = //(1.f / (*rest_density)) *
					(lambda_i + lambda[original_index] +scorr)*
					gradient;
				
				correction += res;
			}
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

// compute correction from boundary particles
inline __device__
float3 pbf_correction_boundary(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	lambda_i,
	float*	rest_density,
	// boundary
	CellData b_cell_data,
	float*	b_lambda,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = b_cell_data.cellStart[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = b_cell_data.grid_index[j];

				float lambda_j = b_lambda[original_index];
				float3 pos2 = b_cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradient = Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

				float scorr = -0.1f;
				float x = Poly6_W_CUDA(dist, params.effective_radius) /
					Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
				x = pow(x, 4);
				scorr = scorr * x * dt * dt;

				//printf("scorr: %f\n", scorr);

				float3 res = //(1.f / (*rest_density)) *
					(lambda_i + lambda_j) *// +scorr)*
					gradient;

				correction += res;
			}
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
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
	//boundary
	CellData cell_data,
	float*	b_volume,
	uint	numParticles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];
	
	// initial density
	float rho = mass[originalIndex] * Poly6_W_CUDA(0, params.effective_radius);

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// traverse 27 neighbors (fluid - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				rho += pbf_density_0(
					neighbor_pos, index, 
					pos, sorted_pos, mass, 
					rest_density,
					cellStart, cellEnd, gridParticleIndex
				);
			}
		}
	}

	// use gridPos to traverse 27 surrounding grids (fluid - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_boundary(
					// fluid
					neighbor_gridPos,
					pos, 
					rest_density,
					// boundary
					b_volume,
					cell_data
				);
			}
		}
	}


	// Update date density and constraint value
	density[originalIndex] = rho;
	C[originalIndex] = (rho / (*rest_density)) - 1.f;

	//printf("rho = %f\n", rho);
	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);

}

__global__
void compute_boundary_density_d(
	// fluid
	float*		rest_density,				// input: rest density
	float3*		sorted_pos,					// input: sorted pos of fluid particle
	float*		mass,						// input: mass of fluid paritcle
	uint*		cellStart,
	uint*		cellEnd,
	uint*		gridParticleIndex,			// input: sorted particle indices (for original_index of fluid particles)
	// boundary
	CellData	b_cell_data,
	float*		b_mass,
	float*		b_volume,
	float*		b_C,
	float*		b_density,					// output: boundary density
	uint		b_numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= b_numParticles) return;

	// original index of boundary particle
	uint originalIndex = b_cell_data.grid_index[index];

	// read position from sorted arrays
	float3 pos = b_cell_data.sorted_pos[index];

	// initial density 
	float rho = (*rest_density) * b_volume[originalIndex] * Poly6_W_CUDA(0, params.effective_radius);

	// get address in grid of boundary particles (basically the same as fluid particle)
	int3 gridPos = calcGridPos(pos);

	// use gridPos to traverse 27 surrounding grids (boundary - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_1(
					neighbor_gridPos, index,
					pos, b_cell_data.sorted_pos,
					b_mass,
					rest_density,
					b_cell_data.cellStart,
					b_cell_data.cellEnd,
					b_cell_data.grid_index,
					b_volume
				);
			}
		}
	}

	// use gridPos to traverse 27 surrounding grids (boundary - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density(
					// boundary
					neighbor_gridPos,
					pos,
					// fluid
					mass,
					sorted_pos,
					cellStart,
					cellEnd,
					gridParticleIndex
				);
			}
		}
	}

	// Update density of fluid particle
	b_density[originalIndex] = rho;
	// **repeated code**
	// Recompute constraint value of fluid particle
	b_C[originalIndex] = (b_density[originalIndex] / (*rest_density)) - 1.f;
}

/* fluid - boundary */
__global__
void compute_lambdas_d(
	float*	lambda,						// output: computed density
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	float*	C,							// input: contraint
	float*  mass,
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	CellData cell_data,
	
	float*	b_volume,
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

	const float epsilon = 100.f;
	float3 gradientC_i = make_float3(0);
		//-(1.f / (*rest_density)) *
		//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);
	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_0(
					neighbor_pos, index,
					pos, rest_density,
					mass, sorted_pos,
					cellStart, cellEnd, 
					gridParticleIndex
				);
				gradientC_sum += res;
			}
		}
	}

	// traverse 27 neighbors in "boundary cells"
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_boundary(
					neighbor_pos, 
					pos, rest_density,
					mass[originalIndex],  // paritcle_mass
					cell_data,
					b_volume
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	lambda[originalIndex] /= gradientC_sum + epsilon;

	//lambda[originalIndex] = lambda_res;
}

__global__
void compute_boundary_lambdas_d(
	float* b_lambda,				// lambda of boundary particles
	float* b_vol,
	float3* b_pos,
	float* b_C,
	float* b_mass,
	CellData b_cell_data,
	// Cell data of fluid particles
	float3* sorted_pos,
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	float*	rest_density,
	uint	b_numParticles		// number of boundary particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= b_numParticles) return;

	uint originalIndex = b_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = b_cell_data.sorted_pos[index];

	// initial density
	b_lambda[originalIndex] = -b_C[originalIndex];
	float particle_mass = b_mass[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	const float epislon = 0.001f;
	float3 gradientC_i = make_float3(0);
	//-(1.f / (*rest_density)) *
	//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);
	
	// traverse 27 neighbors in boundary cells (boundary - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_1(
					neighbor_pos, index,
					pos, rest_density,
					b_mass,
					b_cell_data.sorted_pos,
					b_cell_data.cellStart, b_cell_data.cellEnd, b_cell_data.grid_index,
					b_vol
				);
				gradientC_sum += res;
			}
		}
	}

	// traverse 27 neighbors in "fluid cells" (boundary - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_boundary_lambda(
					// boundary
					neighbor_pos,
					pos, rest_density,
					particle_mass,			// paritcle_mass
					b_vol[originalIndex],	// volume
					// fluid
					sorted_pos,
					cellStart,
					cellEnd,
					gridParticleIndex
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	b_lambda[originalIndex] /= gradientC_sum + epislon;

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
	// boundary
	CellData b_cell_data,
	float*	b_lambda,
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

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				correction += pbf_correction_boundary(
					neighbor_pos,
					index,
					pos,
					lambda_i,
					rest_density,
					b_cell_data,
					b_lambda,
					dt
				);
			}
		}
	}
	correction = (1.f / (*rest_density)) * correction;
	//compute new position
	new_pos[originalIndex] = pos + correction;
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
	//predict_pos[index] = t_pos;
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
	thrust::sort_by_key(
		thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex)
	);
}

void sort_particles_boundary(CellData cell_data, uint numParticles)
{
	thrust::sort_by_key(
		thrust::device_ptr<uint>(cell_data.grid_hash),
		thrust::device_ptr<uint>(cell_data.grid_hash + numParticles),
		thrust::device_ptr<uint>(cell_data.grid_index)
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

void solve_sph_fluid(
	float3* pos,
	float3* new_pos,
	float3* predict_pos,
	float3* vel,
	float3* sorted_pos,
	float3* sorted_vel,
	float* mass,
	float* density,
	float* rest_density,
	float* C,
	float* lambda,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint	numParticles,
	uint	numCells,
	CellData b_cell_data,
	float3*	 b_pos,
	float*	 b_mass,
	float*   b_volume,
	float*	 b_C,
	float*	 b_density,
	float*	 b_lambda,
	uint     b_num_particles,
	float	dt)
{
	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	for (int i = 0; i < 1; ++i)
	{
		// CUDA SPH Kernel
		// compute density
		t1 = std::chrono::high_resolution_clock::now();
		compute_density_d << <numBlocks, numThreads >> > (
			density, rest_density,
			sorted_pos,
			mass, C,
			gridParticleIndex,
			cellStart,
			cellEnd,
			b_cell_data,
			b_volume,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_density_d ");
		// compute density contributed by boundary particles
		compute_boundary_density_d << <numBlocks, numThreads >> > (
			rest_density,
			sorted_pos,
			mass,
			cellStart,
			cellEnd,
			gridParticleIndex,
			b_cell_data,
			b_mass,
			b_volume,
			b_C,
			b_density,
			b_num_particles
			);
		// compute density of bounary particles
		// compute_boundary_density_d();
		getLastCudaError("Kernel execution failed: compute_density_boundary_d ");
		t2 = std::chrono::high_resolution_clock::now();
		// compute lambda
 		compute_lambdas_d << <numBlocks, numThreads >> > (
			lambda,
			rest_density,
			sorted_pos,
			C,
			mass,
			gridParticleIndex,
			cellStart,
			cellEnd,
			b_cell_data,
			b_volume,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_lambdas_d ");
		compute_boundary_lambdas_d << <numBlocks, numThreads >> > (
			b_lambda,
			b_volume,
			b_pos,
			b_C,
			b_mass,
			b_cell_data,
			sorted_pos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			rest_density,
			b_num_particles
		);
		getLastCudaError("Kernel execution failed: compute_boundary_lambdas_d ");
		t3 = std::chrono::high_resolution_clock::now();
		// compute new position
		compute_position_correction << <numBlocks, numThreads >> > (
			lambda,
			rest_density,
			sorted_pos,
			new_pos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			b_cell_data,
			b_lambda,
			numParticles,
			dt
			);
		getLastCudaError("Kernel execution failed: compute_position_correction ");
		// correct this iteration
		apply_correction << <numBlocks, numThreads >> > (new_pos, predict_pos, numParticles);
		getLastCudaError("Kernel execution failed: apply_correction ");

		t4 = std::chrono::high_resolution_clock::now();
	}
	// final correction
	finalize_correction << <numBlocks, numThreads >> > (
		pos, new_pos, predict_pos, vel, 
		numParticles, 
		dt
	);
	getLastCudaError("Kernel execution failed: finalize_correction ");
	t5 = std::chrono::high_resolution_clock::now();
	{
		ImGui::Begin("CUDA Performance");
		ImGui::Text("Density:     %.5lf (ms)", (t2 - t1).count() / 1000000.0f);
		ImGui::Text("Lambda:      %.5lf (ms)", (t3 - t2).count() / 1000000.0f);
		ImGui::Text("Correction:  %.5lf (ms)", (t4 - t3).count() / 1000000.0f);
		ImGui::Text("Finalize:    %.5lf (ms)", (t5 - t4).count() / 1000000.0f);
		ImGui::End();
	}

}

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions)
{
	test_offset << <block_num, thread_num >> > (positions);
}
