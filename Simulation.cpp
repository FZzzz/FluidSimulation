#include "Simulation.h"
#include "imgui/imgui.h"
#include "CollisionDetection.h"
#include "SPHKernel.h"
#include <iostream>
#include "imgui/imgui.h"
#include <omp.h>

Simulation::Simulation()
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr), 
	m_initialized(false), m_world_desc(SimWorldDesc(-9.8f, 0.f)), m_pause(true),
	m_rest_density(1.2f)
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
	for (uint32_t i = 0; i < m_solver->getSolverIteration(); ++i)
	{		
		ComputeDensity(effective_radius);
		ComputeLambdas(effective_radius);
		ComputeSPHParticlesCorrection(effective_radius);

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
		particles->m_new_positions[i] = particles->m_positions[i] + dt * particles->m_velocity[i];
	}
	/*
	for (auto p : m_particle_system->getParticles())
	{
		// external forces
		p->m_force = p->m_mass * glm::vec3(0, m_world_desc.gravity, 0);

		p->m_velocity = p->m_velocity + dt * p->m_massInv * p->m_force;
		
		// dampVelocity()

		p->m_new_position = p->m_position + dt * p->m_velocity;

		// Update colliders
		p->UpdateCollider();
	}
	*/
}

void Simulation::FindNeighborParticles(float effective_radius)
{
	m_neighbor_searcher->NaiveSearch(effective_radius);
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

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));
			particles->m_density[i] = particles->m_mass[i] * SPHKernel::Poly6_W(0, effective_radius);


			for (int j = 0; j < neighbors.size(); ++j)
			{
				float distance = glm::distance(particles->m_new_positions[i], particles->m_new_positions[neighbors[j]]);
				particles->m_density[i] += particles->m_mass[neighbors[j]] * SPHKernel::Poly6_W(distance, effective_radius);
			}
			particles->m_C[i] = particles->m_density[i] / m_rest_density - 1.f;
		}
	}
	/*
	auto& particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles.size(); ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));
			particles[i]->m_density = particles[i]->m_mass * SPHKernel::Poly6_W(0, effective_radius);


			for (int j = 0; j < neighbors.size(); ++j)
			{
				float distance = glm::distance(particles[i]->m_new_position, particles[neighbors[j]]->m_new_position);
				particles[i]->m_density += particles[neighbors[j]]->m_mass * SPHKernel::Poly6_W(distance, effective_radius);
			}
			particles[i]->m_C = particles[i]->m_density / m_rest_density - 1.f;
		}
	}
	*/
}

void Simulation::ComputeLambdas(float effective_radius)
{
	const float epsilon = 1.0e-6f;
	/* Compute density constraints */
	ParticleSet* const particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared)
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
				glm::vec3 diff = particles->m_new_positions[i] - particles->m_new_positions[neighbors[j]];
				float distance = glm::distance(particles->m_new_positions[i], particles->m_new_positions[neighbors[j]]);

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
	/*
	auto& particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles.size(); ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			// Reset Lagragian multiplier
			particles[i]->m_lambda = -particles[i]->m_C;
			glm::vec3 gradientC_i = glm::vec3(0, 0, 0);
			float gradientC_sum = 0.f;

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles[i]->m_new_position - particles[neighbors[j]]->m_new_position;
				float distance = glm::distance(particles[i]->m_new_position, particles[neighbors[j]]->m_new_position);

				glm::vec3 gradientC_j = (1.f / m_rest_density) * SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				float dot_value = glm::dot(gradientC_j, gradientC_j);

				gradientC_i += gradientC_j;
				gradientC_sum += dot_value;
			}
			float dot_value = glm::dot(gradientC_i, gradientC_i);
			gradientC_sum += dot_value;
			particles[i]->m_lambda /= gradientC_sum + epsilon;
		}
	}
	*/
}

void Simulation::ComputeSPHParticlesCorrection(float effective_radius)
{
	ParticleSet* const particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles->m_new_positions[i] - particles->m_new_positions[neighbors[j]];
				float distance = glm::distance(particles->m_new_positions[i], particles->m_new_positions[neighbors[j]]);

				glm::vec3 result = (1.f / m_rest_density) *
					(particles->m_lambda[i] + particles->m_lambda[neighbors[j]]) *
					SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				particles->m_new_positions[i] += result;
			}
		}
	}
	/*
	auto& particles = m_particle_system->getParticles();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles.size(); ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles[i]->m_new_position - particles[neighbors[j]]->m_new_position;
				float distance = glm::distance(particles[i]->m_new_position, particles[neighbors[j]]->m_new_position);

				glm::vec3 result = (1.f / m_rest_density) *
					(particles[i]->m_lambda + particles[neighbors[j]]->m_lambda) *
					SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				particles[i]->m_new_position += result;
			}
		}
	}
	*/
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


	/*
	auto& particles = m_particle_system->getParticles();
	for (size_t i = 0; i < particles.size(); ++i)
	{
		const glm::vec3& point_pos = particles[i]->m_new_position;

		for (size_t j = 0; j < m_colliders.size(); ++j)
		{
			if (particles[i]->TestCollision(m_colliders[j]))
				particles[i]->OnCollision(m_colliders[j], dt);
		}
	}
	*/

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
	/*
	for(auto p : m_particle_system->getParticles())
	{
		p->Update(dt);
	}
	*/
	m_particle_system->getParticles()->Update(dt);
	m_particle_system->Update();
/*
#if _DEBUG	
	std::cout << particles[0]->position.x << " "
		<< particles[1]->position.x << std::endl;
#endif
*/
/*
#ifdef _DEBUG
	{
		ImGui::Begin("Particles");
		float p0[3] = { particles[0]->position.x,
						particles[0]->position.y,
						particles[0]->position.z };
		ImGui::InputFloat3("P0 position", p0, 5, ImGuiInputTextFlags_ReadOnly);

		float p1[3] = { particles[1]->position.x,
						particles[1]->position.y,
						particles[1]->position.z };
		ImGui::InputFloat3("P1 position", p1, 5, ImGuiInputTextFlags_ReadOnly);
		ImGui::End();
	}
#endif
*/

}
