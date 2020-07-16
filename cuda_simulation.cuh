#ifndef _CUDA_SIMULATION_CUH_
#define _CUDA_SIMULATION_CUH_

#include <cuda_runtime.h>
#include "Particle.h"
#include "Simulation.h"
//#include <helper_math.h>

#define MAX_THREAD_NUM 512

// simulation parameters
__constant__ SimParams params;

void allocateArray(void** devPtr, size_t size);

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions);

void setParams(SimParams* param_in);

void integrate(
    float3* pos,
    float3* vel,
    float deltaTime,
    uint numParticles
);

void integratePBD(
    float3* pos, float3* vel,
    float3* force, float* massInv,
    float3* predict_pos, float3* new_pos,
    float deltaTime,
    uint numParticles
);

void compute_grid_size(uint n, uint block_size, uint& num_blocks, uint& num_threads);

void calculate_hash(
    uint* grid_particle_hash,
    uint* grid_particle_index,
    float3* pos,
    uint    num_particles
);

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
    uint	numCells
);

void sort_particles(
    uint* dGridParticleHash, uint* dGridParticleIndex,
    uint numParticles
);

void solve_dem_collision(
    float3* newVel,
    float3* sortedPos,
    float3* sortedVel,
    uint* gridParticleIndex,
    uint* cellStart,
    uint* cellEnd,
    uint   numParticles,
    uint   numCells,
    float dt
);

void compute_rest_density(
    float* rest_density,					// output: computed density
    float3* sorted_pos,				// input: sorted mass
    float* mass,					// input: mass
    uint* gridParticleIndex,		// output: sorted particle indices
    uint* cellStart,
    uint* cellEnd,
    uint  numParticles
);

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
    float	dt
);


#endif