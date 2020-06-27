#ifndef _CUDA_SIMULATION_CUH_
#define _CUDA_SIMULATION_CUH_

#include <cuda_runtime.h>
#include <helper_math.h>

#define MAX_THREAD_NUM 512

struct SimParams
{
    float3 collider_pos;
    float  collider_radius;

    float3 gravity;
    float global_damping;
    float particle_radius;

    uint3 grid_size;
    uint num_cells;
    float3 world_origin;
    float3 cell_size;
    /*
    uint num_bodies;
    uint max_particles_per_cell;
    */
    float spring;
    float damping;
    float shear;
    float attraction;
    float boundary_damping;
};

// simulation parameters
__constant__ SimParams params;

void allocateArray(void** devPtr, size_t size);

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions);

void setParams(SimParams* param_in);

void integrate(
    float3* pos,
    float3* vel,
    float deltaTime,
    uint numParticles);

void compute_grid_size(uint n, uint block_size, uint& num_blocks, uint& num_threads);

void calculate_hash(
    uint* grid_particle_hash,
    uint* grid_particle_index,
    float3* pos,
    uint    num_particles);

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
    uint	numCells);

void sort_particles(
    uint* dGridParticleHash, uint* dGridParticleIndex,
    uint numParticles);

void collide(
    float3* newVel,
    float3* sortedPos,
    float3* sortedVel,
    uint* gridParticleIndex,
    uint* cellStart,
    uint* cellEnd,
    uint   numParticles,
    uint   numCells);

#endif