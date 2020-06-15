#include "Simulation.h"
#include "imgui/imgui.h"
#include "CollisionDetection.h"
#include "SPHKernel.h"
#include <iostream>
#include "imgui/imgui.h"
#include <omp.h>

Simulation::Simulation()
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr),
	m_initialized(false), m_world_desc(SimWorldDesc(-9.8f, 0.f)), m_pause(false),
	m_first_frame(false),
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
	m_solver = std::make_shared<ConstraintSolver>(mode);
	m_particle_system = particle_system;
	m_neighbor_searcher = std::make_shared<NeighborSearch>(m_particle_system);
	m_neighbor_searcher->Initialize();
	m_initialized = true;
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

	PredictPositions(dt);

	CollisionDetection(dt);
	HandleCollisionResponse();
	GenerateCollisionConstraint();

	FindNeighborParticles(effective_radius);

	
	if (!m_first_frame)
		ComputeRestDensity();
	
	//m_rest_density = 10.f / 12.f;

	for (uint32_t i = 0; i < m_solver->getSolverIteration(); ++i)
	{		
		ComputeDensity(effective_radius);
		ComputeLambdas(effective_radius);
		ComputeSPHParticlesCorrection(effective_radius, dt);
		UpdatePredictPosition();
	}
	
 	if (!ProjectConstraints(dt))
		return false;

	ApplySolverResults(dt);

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
	
	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
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
	m_first_frame = true;
}

void Simulation::Pause()
{
	m_pause = !m_pause;
}

void Simulation::setGravity(float gravity)
{
	m_world_desc.gravity = gravity;
}

void Simulation::PredictPositions(float dt)
{
	/*
	 * Update position and velocity
	 * 1. forall vertices i do v_i = v_i + dt * w_i * f_{ext}
	 * 2. damp velocities 	
	 */
	ParticleSet* particles = m_particle_system->getParticles();
	for (size_t i = 0; i < particles->m_size; ++i)
	{
		particles->m_force[i] = particles->m_mass[i] * glm::vec3(0, m_world_desc.gravity, 0);
		particles->m_velocity[i] = particles->m_velocity[i] + dt * particles->m_massInv[i] * particles->m_force[i];
		particles->m_predict_positions[i] = particles->m_positions[i] + dt * particles->m_velocity[i];
		particles->m_new_positions[i] = particles->m_predict_positions[i];
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
			glm::vec3 gradientC_i = glm::vec3(0, 0, 0);
			float gradientC_sum = 0.f;

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles->m_predict_positions[i] - particles->m_predict_positions[neighbors[j]];
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);

				glm::vec3 gradientC_j = (1.f / m_rest_density) * SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				float dot_value = glm::dot(gradientC_j, gradientC_j);

				gradientC_i += gradientC_j;
				gradientC_sum += dot_value;
			}
			float dot_value = glm::dot(gradientC_i, gradientC_i);
			gradientC_sum += dot_value;
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
