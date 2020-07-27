#include "Simulation.h"
#include "imgui/imgui.h"
#include "CollisionDetection.h"
#include "SPHKernel.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <omp.h>
#include <chrono>
#include "cuda_simulation.cuh"


Simulation::Simulation()
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr),
	m_initialized(false), m_world_desc(SimWorldDesc(-9.8f, 0.f)), m_pause(true),
	m_first_frame(true),
	m_rest_density(0.8f)
{
	

}
Simulation::Simulation(SimWorldDesc desc)
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr), 
	m_initialized(false), m_world_desc(desc), m_pause(false)
{
}

Simulation::~Simulation()
{
}

void Simulation::Initialize(PBD_MODE mode, std::shared_ptr<ParticleSystem> particle_system)
{
	m_particle_system = particle_system;
	
	uint3 grid_size = make_uint3(64, 64, 64);
	
	m_neighbor_searcher = std::make_shared<NeighborSearch>(m_particle_system, grid_size);
	m_solver = std::make_shared<ConstraintSolver>(mode);

	SetupSimParams();
	GenerateFluidCube();
		
	m_neighbor_searcher->InitializeCUDA();
	m_initialized = true;
	
#ifdef _USE_CUDA_
	cudaMalloc((void**)&m_d_rest_density, sizeof(float));
	cudaMemcpy((void*)m_d_rest_density, (void*)&m_rest_density, sizeof(float), cudaMemcpyHostToDevice);
#endif


}

/*
 * Step simulation
 * @param[in] dt time step
 * @retval true  Successfully project constraints
 * @retval false Failed on constraint projection
 */
bool Simulation::Step(float dt)
{
	if (!m_initialized)
		return false;

	if (m_pause)
		return true;

	const float effective_radius = 1.f;
	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;

	PredictPositions(dt);

	t1 = std::chrono::high_resolution_clock::now();
	CollisionDetection(dt);
	HandleCollisionResponse();
	GenerateCollisionConstraint();

	t2 = std::chrono::high_resolution_clock::now();
	FindNeighborParticles(effective_radius);

	t3 = std::chrono::high_resolution_clock::now();
	
	if (m_first_frame)
	{
		ComputeRestDensity();
		std::cout << "Rest density: " << m_rest_density << std::endl;
	}
	
	
	//m_rest_density = 10.f / 12.f;

	for (uint32_t i = 0; i < m_solver->getSolverIteration(); ++i)
	{		
		ComputeDensity(effective_radius);
		ComputeLambdas(effective_radius);
		ComputeSPHParticlesCorrection(effective_radius, dt);
		UpdatePredictPosition();
	}
	t4 = std::chrono::high_resolution_clock::now();
	
 	if (!ProjectConstraints(dt))
		return false;

	ApplySolverResults(dt);

	{
		ImGui::Begin("Performance");
		ImGui::Text("Collision:\t %.5lf (ms)", (t2 - t1).count() / 1000000.0);
		ImGui::Text("Searching:\t %.5lf (ms)", (t3 - t2).count() / 1000000.0);
		ImGui::Text("Correction:\t%.5lf (ms)", (t4 - t3).count() / 1000000.0);
		ImGui::Text("GL update:\t%.5lf (ms)", m_particle_system->getUpdateTime());
		ImGui::End();
	}

	size_t glm_size = sizeof(glm::vec3);
	size_t cu_size = sizeof(float3);

	//m_pause = true;

	return true;
}

bool Simulation::StepCUDA(float dt)
{
	if (!m_initialized)
		return false;

	if (m_pause)
		return true;
	
	const float effective_radius = 1.f;
	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;

	ParticleSet* particles = m_particle_system->getParticles();
	cudaGraphicsResource** vbo_resource = m_particle_system->getCUDAGraphicsResource();
	glm::vec3* positions = particles->m_positions.data();
	unsigned int numParticles = particles->m_size;

	// Map vbo to m_d_positinos
	cudaGraphicsMapResources(1, vbo_resource, 0);

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&(particles->m_d_positions), &num_bytes, *vbo_resource);
	
	// Integrate
	//integrate(particles->m_d_positions, particles->m_d_velocity, dt, particles->m_size);
	
	t1 = std::chrono::high_resolution_clock::now();
	integratePBD(
		particles->m_d_positions, particles->m_d_velocity,
		particles->m_d_force, particles->m_d_massInv,
		particles->m_d_predict_positions, particles->m_d_new_positions,
		dt,
		numParticles
		);
	
	t2 = std::chrono::high_resolution_clock::now();
	// Neighbor search
	calculate_hash(
		m_neighbor_searcher->m_d_grid_particle_hash,
		m_neighbor_searcher->m_d_grid_particle_index,
		//particles->m_d_positions,
		particles->m_d_predict_positions,
		numParticles
	);
	sort_particles(
		m_neighbor_searcher->m_d_grid_particle_hash,
		m_neighbor_searcher->m_d_grid_particle_index,
		numParticles
	);
	reorderDataAndFindCellStart(
		m_neighbor_searcher->m_d_cellStart,
		m_neighbor_searcher->m_d_cellEnd,
		particles->m_d_sorted_position,
		particles->m_d_sorted_velocity,
		m_neighbor_searcher->m_d_grid_particle_hash,
		m_neighbor_searcher->m_d_grid_particle_index,
		//particles->m_d_positions,
		particles->m_d_predict_positions,
		particles->m_d_velocity,
		numParticles,
		m_neighbor_searcher->m_num_grid_cells
	);
	t3 = std::chrono::high_resolution_clock::now();

	if (m_first_frame)
	{
		/*
		compute_rest_density(
			m_d_rest_density,
			particles->m_d_sorted_position,
			particles->m_d_mass,
			m_neighbor_searcher->m_d_grid_particle_index,
			m_neighbor_searcher->m_d_cellStart,
			m_neighbor_searcher->m_d_cellEnd,
			numParticles
		);
		//cudaThreadSynchronize();

		cudaMemcpy(&m_rest_density, m_d_rest_density, sizeof(float), cudaMemcpyDeviceToHost);
		*/
		//cudaMemcpy((void*)m_d_rest_density, &m_rest_density, sizeof(float), cudaMemcpyHostToDevice);
		std::cout << "Rest density: " << m_rest_density << std::endl;

		m_first_frame = false;
	}
	/*
	// Solve dem particles collision
	solve_dem_collision(
		particles->m_d_velocity,
		particles->m_d_sorted_position,
		particles->m_d_sorted_velocity,
		m_neighbor_searcher->m_d_grid_particle_index,
		m_neighbor_searcher->m_d_cellStart,
		m_neighbor_searcher->m_d_cellEnd,
		numParticles,
		m_neighbor_searcher->m_num_grid_cells,
		dt
	);
	*/
	
	solve_sph_fluid(
		particles->m_d_positions,
		particles->m_d_new_positions, 
		particles->m_d_predict_positions,
		particles->m_d_velocity,
		particles->m_d_sorted_position, particles->m_d_sorted_velocity,
		particles->m_d_mass,
		particles->m_d_density, m_d_rest_density,
		particles->m_d_C,
		particles->m_d_lambda,
		m_neighbor_searcher->m_d_grid_particle_index,
		m_neighbor_searcher->m_d_cellStart,
		m_neighbor_searcher->m_d_cellEnd,
		numParticles,
		m_neighbor_searcher->m_num_grid_cells,
		dt
	);
	
	t4 = std::chrono::high_resolution_clock::now();

	{
		ImGui::Begin("CUDA Performance");
		ImGui::Text("Integrate:   %.5lf (ms)", (t2 - t1).count() / 1000000.0f);
		ImGui::Text("Search:      %.5lf (ms)", (t3 - t2).count() / 1000000.0f);
		//ImGui::Text("Solve:       %.5lf (ms)", (t4 - t2).count() / 1000000.0f);
		ImGui::End();
	}
	
	// Unmap CUDA buffer object
	cudaGraphicsUnmapResources(1, vbo_resource, 0);

	//m_pause = true;

	return true;
}

void Simulation::AddCollider(Collider* collider)
{
	m_colliders.push_back(collider);
	m_collision_table.push_back(std::vector<Collider*>());
}

void Simulation::AddStaticConstraint(Constraint* constraint)
{
	m_static_constraints.push_back(constraint);
}

void Simulation::AddStaticConstraints(std::vector<Constraint*> constraints)
{
	m_static_constraints.insert(m_static_constraints.end(), constraints.begin(), constraints.end());
}

void Simulation::SetSolverIteration(uint32_t iter_count)
{
	m_solver->setSolverIteration(iter_count);
}

void Simulation::ComputeRestDensity()
{
	const float effective_radius = 1.f;
	ParticleSet* const particles = m_particle_system->getParticles();

	m_rest_density = 0;
	
	//#pragma omp parallel default(shared) num_threads(8)
	{
		//#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));
			particles->m_density[i] = particles->m_mass[i] * SPHKernel::Poly6_W(0, effective_radius);


			for (int j = 0; j < neighbors.size(); ++j)
			{
				float distance = glm::distance(particles->m_positions[i], particles->m_positions[neighbors[j]]);
				particles->m_density[i] += particles->m_mass[neighbors[j]] * SPHKernel::Poly6_W(distance, effective_radius);
			}
			m_rest_density += particles->m_density[i];
		}
	}

	m_rest_density /= static_cast<float>(particles->m_size);
	m_first_frame = false;
}

void Simulation::Pause()
{
	m_pause = !m_pause;
}

void Simulation::setGravity(float gravity)
{
	m_world_desc.gravity = gravity;
}

void Simulation::SetupSimParams()
{
	//const size_t n_particles = 1000;
	const float particle_mass = 0.05f;
	const float n_kernel_particles = 20.f;	
	// water density = 1000 kg/m^3
	m_rest_density = 1000.f; 
	m_particle_mass = particle_mass;

	float effective_radius, particle_radius;
	
	/* Compute parameters from mass and n_particles*/
	m_volume = n_kernel_particles * particle_mass / m_rest_density;
	effective_radius = powf(((3.0f * m_volume) / (4.0f * M_PI)), 1.0f / 3.0f);
	particle_radius = powf((M_PI / (6.0f * n_kernel_particles)), 1.0f / 3.0f) * effective_radius;

	std::cout << "Particle mass: " << particle_mass << std::endl;
	std::cout << "Effective radius: " << effective_radius << std::endl;
	std::cout << "Particle radius: " << particle_radius << std::endl;

	m_sim_params = new SimParams();
	m_sim_params->gravity = make_float3(0.f, -9.81f, 0.f);
	m_sim_params->global_damping = 0.99f;
	m_sim_params->particle_radius = particle_radius;
	m_sim_params->effective_radius = effective_radius;
	m_sim_params->grid_size = m_neighbor_searcher->m_grid_size;
	m_sim_params->num_cells = m_neighbor_searcher->m_num_grid_cells;
	m_sim_params->world_origin = make_float3(0, 0, 0);
	m_sim_params->cell_size = make_float3(m_sim_params->effective_radius);
	m_sim_params->spring = 0.5f;
	m_sim_params->damping = 0.02f;
	m_sim_params->shear = 0.1f;
	m_sim_params->attraction = 0.0f;
	m_sim_params->boundary_damping = 1.0f;

	setParams(m_sim_params);
}

void Simulation::InitializeBoundaryParticles()
{
	const float diameter = 2.f * m_sim_params->particle_radius;
	// number of particles on x,y,z
	int nx, ny, nz;
	size_t n_particles = 0;
	// fluid cube extends
	// buttom
	float x, y, z;

	size_t n_particles = 8 * nx * ny * nz;

	// Initialize boundary particles
	ParticleSet* particles = m_particle_system->AllocateParticles(n_particles, m_particle_mass);
	// Initialize positions
	// Buttom boundary
	size_t idx = 0;
	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				//int idx = k + j * 10 + i * 100;
				x = 0.f + diameter * static_cast<float>(i);
				y = 1.51f + diameter * static_cast<float>(j);
				z = -0.f + diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				particles->m_positions[idx] = pos;
				particles->m_new_positions[idx] = pos;
				particles->m_predict_positions[idx] = pos;
				idx++;
			}
		}
	}
	// Left boundary
	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				//int idx = k + j * 10 + i * 100;
				x = 0.f + diameter * static_cast<float>(i);
				y = 1.51f + diameter * static_cast<float>(j);
				z = -0.f + diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				particles->m_positions[idx] = pos;
				particles->m_new_positions[idx] = pos;
				particles->m_predict_positions[idx] = pos;
				idx++;
			}
		}
	}
	// Right boundary
	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				//int idx = k + j * 10 + i * 100;
				x = 0.f + diameter * static_cast<float>(i);
				y = 1.51f + diameter * static_cast<float>(j);
				z = -0.f + diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				particles->m_positions[idx] = pos;
				particles->m_new_positions[idx] = pos;
				particles->m_predict_positions[idx] = pos;
				idx++;
			}
		}
	}
	// Precompute hash
	// Sort
	// Reorder

}

void Simulation::GenerateFluidCube()
{
	// diameter of particle
	const float diameter = 2.f * m_sim_params->particle_radius;
	// number of particles on x,y,z
	int nx, ny, nz;
	// fluid cube extends
	glm::vec3 half_extend(0.5f, 1.5f, 0.5f);
	
	nx = static_cast<int>(half_extend.x / diameter) - 1;
	ny = static_cast<int>(half_extend.y / diameter) - 1;
	nz = static_cast<int>(half_extend.z / diameter) - 1;

	float x, y, z;
	
	//const float diameter = 0.5f;

	size_t n_particles = 8 * nx * ny * nz;
	ParticleSet* particles = m_particle_system->AllocateParticles(n_particles, m_particle_mass);

	std::cout << "n_particles: " << n_particles << std::endl;
	// set positions
	size_t idx = 0;
	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				//int idx = k + j * 10 + i * 100;
				x = 0.f + diameter * static_cast<float>(i);
				y = 1.51f + diameter * static_cast<float>(j);
				z = -0.f + diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				particles->m_positions[idx] = pos;
				particles->m_new_positions[idx] = pos;
				particles->m_predict_positions[idx] = pos;
				idx++;
			}
		}
	}
#ifdef _USE_CUDA_
	m_particle_system->InitializeCUDA();
#else
	m_particle_system->Initilize();
#endif
}

void Simulation::PredictPositions(float dt)
{
	/*
	 * Update position and velocity
	 * 1. forall vertices i do v_i = v_i + dt * w_i * f_{ext}
	 * 2. damp velocities 	
	 */
	ParticleSet* particles = m_particle_system->getParticles();

#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < particles->m_size; ++i)
		{
			particles->m_force[i] = particles->m_mass[i] * glm::vec3(0, m_world_desc.gravity, 0);
			particles->m_velocity[i] = particles->m_velocity[i] + dt * particles->m_massInv[i] * particles->m_force[i];
			particles->m_predict_positions[i] = particles->m_positions[i] + dt * particles->m_velocity[i];
			particles->m_new_positions[i] = particles->m_predict_positions[i];
		}
	}

}

void Simulation::FindNeighborParticles(float effective_radius)
{
	//m_neighbor_searcher->NaiveSearch(effective_radius);
	m_neighbor_searcher->SpatialSearch(effective_radius);
}

void Simulation::ComputeDensity(float effective_radius)
{
	/*
	{
		const auto& neighbors = m_neighbor_searcher->FetchNeighbors(0);
		ImGui::Begin("Neighbor test");
		ImGui::Text("Object Array Size: %u", neighbors.size());
		ImGui::End();
	}
	*/
	ParticleSet* const particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));
			particles->m_density[i] = particles->m_mass[i] * SPHKernel::Poly6_W(0, effective_radius);


			for (int j = 0; j < neighbors.size(); ++j)
			{
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);
				
				float res = SPHKernel::Poly6_W(distance, effective_radius);
				particles->m_density[i] += particles->m_mass[neighbors[j]] * res;
			}
			particles->m_C[i] = particles->m_density[i] / m_rest_density - 1.f;
			std::cout << "C: " << particles->m_C[i] << std::endl;
		}
	}

}

void Simulation::ComputeLambdas(float effective_radius)
{
	const float epsilon = 1.0e-6f;
	/* Compute density constraints */
	ParticleSet* const particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			// Reset Lagragian multiplier
			particles->m_lambda[i] = -particles->m_C[i];
			glm::vec3 gradientC_i = (1.f / m_rest_density) * SPHKernel::Poly6_W_Gradient(glm::vec3(0), 0, effective_radius);
			float gradientC_sum = glm::dot(gradientC_i, gradientC_i);

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles->m_predict_positions[i] - particles->m_predict_positions[neighbors[j]];
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);

				glm::vec3 gradientC_j = (1.f / m_rest_density) * SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				float dot_value = glm::dot(gradientC_j, gradientC_j);

				//gradientC_i += gradientC_j;
				gradientC_sum += dot_value;
			}
			std::cout << "gradientC_sum: " << gradientC_sum << std::endl;
			//float dot_value = glm::dot(gradientC_i, gradientC_i);
			//gradientC_sum += dot_value;
			particles->m_lambda[i] /= gradientC_sum + epsilon;
		}
	}

}

void Simulation::ComputeSPHParticlesCorrection(float effective_radius, float dt)
{
	ParticleSet* const particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles->m_predict_positions[i] - particles->m_predict_positions[neighbors[j]];
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);
				
				// Artificial pressure
				double scorr = -0.1;
				double x = SPHKernel::Poly6_W(distance, effective_radius) / 
					SPHKernel::Poly6_W(0.3f * effective_radius, effective_radius);
				x = glm::pow(x, 4);
				scorr = scorr * x * dt;

				glm::vec3 result = (1.f / m_rest_density) *
					(particles->m_lambda[i] + particles->m_lambda[neighbors[j]] + static_cast<float>(scorr)) *
					SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				particles->m_new_positions[i] += result;
			}
		}
	}
}

void Simulation::UpdatePredictPosition()
{
	ParticleSet* const particles = m_particle_system->getParticles();
	for (size_t i = 0; i < particles->m_size; ++i)
	{
		particles->m_predict_positions[i] = particles->m_new_positions[i];
	}
}

void Simulation::CollisionDetection(float dt)
{	
	// Clean previous CD result
	for (auto vec : m_collision_table)
	{
		vec.clear();
	}

	ParticleSet* const particles = m_particle_system->getParticles();
	for (size_t i = 0; i < particles->m_size; ++i)
	{
		for (size_t j = 0; j < m_colliders.size(); ++j)
		{
			if (particles->TestCollision(i, m_colliders[j]))
				particles->OnCollision(i, m_colliders[j], dt);
		}
	}

	// TODO: Change to m_rigidbodies[i], m_rigidbodies[j]
	for (size_t i = 0; i < m_colliders.size(); ++i)
	{
		for (size_t j = i+1; j < m_colliders.size(); ++j)
		{
			/* Record result if there's contact between two objects */
			if (m_colliders[i]->TestCollision(m_colliders[j]))
				m_collision_table[i].push_back(m_colliders[j]);
		}
	}
}

/*
 * This function handles collision response for specific collision pairs.
 * (Particle v.s. Static plane),  (Particle v.s. Static AABB), (Particle v.s Static OBB)
 */
void Simulation::HandleCollisionResponse()
{
}

/*
 * In jelly simulation the only collision is particle hitting the plane or other static BBs.
 * 
*/
void Simulation::GenerateCollisionConstraint()
{
	/* 
	for(pairs : collision_pairs)
	{
		// Generate collision constraint
	}
	*/
}

bool Simulation::ProjectConstraints(const float &dt)
{
	m_solver->SolveConstraints(dt, m_static_constraints, m_collision_constraints);

	return true;
}

void Simulation::AddCollisionConstraint(Constraint* constraint)
{
	m_collision_constraints.push_back(constraint);
}

void Simulation::ApplySolverResults(float dt)
{
	m_particle_system->getParticles()->Update(dt);
	m_particle_system->Update();
}
