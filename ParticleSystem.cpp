#include "ParticleSystem.h"
#include <cuda_runtime.h>
#include <chrono>
#include "imgui/imgui.h"

ParticleSystem::ParticleSystem() :
	m_particles(nullptr),
	//m_particles(nullptr),
	m_point_sprite_size(5.f),
	m_vao(-1),
	m_vbo(-1),
	m_ebo(-1)
{
}

ParticleSystem::~ParticleSystem()
{
	//m_particles.clear();
	m_particle_indices.clear();
}

void ParticleSystem::Initialize()
{
	GenerateParticleGLBuffers();
	UpdateParticleGLBUfferData();
}

void ParticleSystem::InitializeCUDA()
{
	GenerateParticleGLBuffers();
	SetupCUDAMemory();
	UpdateParticleGLBUfferData();
	RegisterCUDAVBO();
}

void ParticleSystem::Update()
{
	std::chrono::steady_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
	//UpdateParticleGLBUfferData();
	t2 = std::chrono::high_resolution_clock::now();
	m_update_elased_time = (t2 - t1).count() / 1000000.0;
}

void ParticleSystem::Release()
{
	if (m_particles == nullptr)
		return;
	cudaFree(m_particles->m_d_prev_positions);
	cudaFree(m_particles->m_d_positions);
	cudaFree(m_particles->m_d_predict_positions);
	cudaFree(m_particles->m_d_new_positions);
	cudaFree(m_particles->m_d_prev_velocity);
	cudaFree(m_particles->m_d_velocity);
	cudaFree(m_particles->m_d_new_velocity);

	cudaFree(m_particles->m_d_sorted_position);
	cudaFree(m_particles->m_d_sorted_velocity);

	cudaFree(m_particles->m_d_force);
	cudaFree(m_particles->m_d_mass);
	cudaFree(m_particles->m_d_massInv);
	cudaFree(m_particles->m_d_density);
	cudaFree(m_particles->m_d_C);
	cudaFree(m_particles->m_d_lambda);

	cudaGraphicsUnregisterResource(m_cuda_vbo_resource);
}

ParticleSet* ParticleSystem::AllocateParticles(size_t n, float particle_mass)
{
	m_particles = new ParticleSet(n, particle_mass);
	for (GLuint i = 0; i < m_particles->m_size; ++i)
	{
		m_particle_indices.push_back(i);
	}
	return m_particles;
}

/*
void ParticleSystem::setParticles(std::vector<std::shared_ptr<Particle>> particles)
{
	m_particle_indices.clear();
	m_particles = particles;
	for (GLuint i = 0; i < particles.size(); ++i)
	{
		m_particle_indices.push_back(i);
	}
	SetupParticleGLBuffers();
}
*/

void ParticleSystem::SetupCUDAMemory()
{
	glm::vec3* prev_positions = m_particles->m_prev_positions.data();
	glm::vec3* positions = m_particles->m_positions.data();
	glm::vec3* predict_positions = m_particles->m_predict_positions.data();
	glm::vec3* new_positions = m_particles->m_new_positions.data();
	
	glm::vec3* velocity = m_particles->m_velocity.data();
	glm::vec3* force = m_particles->m_force.data();

	float* mass = m_particles->m_mass.data();
	float* massInv = m_particles->m_massInv.data();
	float* density = m_particles->m_density.data();
	float* C = m_particles->m_C.data();
	float* lambda = m_particles->m_lambda.data();

	size_t n = m_particles->m_size;

	// Allocate memory space
	cudaMalloc(
		(void**)&(m_particles->m_d_prev_positions), 
		n * sizeof(float3)
	);
	
	/*
	cudaMalloc(
		(void**)&(m_particles->m_d_positions),
		n * sizeof(float3)
	);
	*/
	
	cudaMalloc(
		(void**)&(m_particles->m_d_predict_positions),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_new_positions),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_prev_velocity),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_velocity),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_new_velocity),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_force),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_mass),
		n * sizeof(float)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_massInv),
		n * sizeof(float)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_density),
		n * sizeof(float)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_C),
		n * sizeof(float)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_lambda),
		n * sizeof(float)
	);


	cudaMalloc(
		(void**)&(m_particles->m_d_sorted_position),
		n * sizeof(float3)
	);
	cudaMalloc(
		(void**)&(m_particles->m_d_sorted_velocity),
		n * sizeof(float3)
	);



	// Set value
	cudaMemcpy(
		(void*)m_particles->m_d_prev_positions,
		(void*)prev_positions,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	/*
	cudaMemcpy(
		(void*)m_particles->m_d_positions,
		(void*)positions,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	*/
	
	cudaMemcpy(
		(void*)m_particles->m_d_predict_positions,
		(void*)predict_positions,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_new_positions,
		(void*)new_positions,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_velocity,
		(void*)velocity,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_force,
		(void*)force,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_mass,
		(void*)mass,
		n * sizeof(float),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_massInv,
		(void*)massInv,
		n * sizeof(float),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_density,
		(void*)density,
		n * sizeof(float),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_C,
		(void*)C,
		n * sizeof(float),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_lambda,
		(void*)lambda,
		n * sizeof(float),
		cudaMemcpyHostToDevice
	);

	cudaMemcpy(
		(void*)m_particles->m_d_prev_velocity,
		(void*)velocity,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_new_velocity,
		(void*)velocity,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_sorted_position,
		(void*)positions,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		(void*)m_particles->m_d_sorted_velocity,
		(void*)velocity,
		n * sizeof(float3),
		cudaMemcpyHostToDevice
	);

}

void ParticleSystem::RegisterCUDAVBO()
{
	cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource, m_vbo, cudaGraphicsMapFlagsNone);
}

void ParticleSystem::GenerateParticleGLBuffers()
{
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);
	glGenBuffers(1, &m_ebo);
}

void ParticleSystem::UpdateParticleGLBUfferData()
{
	if (m_particles->m_size == 0)
		return;

	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_particles->m_positions.size(),
		m_particles->m_positions.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	/*
	if (m_particle_indices.size() <= 0)
		return;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_particle_indices.size(),
		m_particle_indices.data(), GL_DYNAMIC_DRAW);
			
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	*/
	glBindVertexArray(0);

	/*
	if(m_cuda_vbo_resource)
		cudaGraphicsUnregisterResource(m_cuda_vbo_resource);
		*/
	RegisterCUDAVBO();
	
	/*
	if (m_particles.size() <= 0)
		return;

	std::vector<glm::vec3> positions;
	for (auto particle : m_particles)
	{
		positions.push_back(particle->m_position);
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions.size(),
		positions.data(), GL_DYNAMIC_DRAW);
	
	glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)offsetof(Particle, m_position));
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	if (m_particle_indices.size() <= 0)
		return;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_particle_indices.size(),
		m_particle_indices.data(), GL_DYNAMIC_DRAW);

	glBindVertexArray(0);
	*/
}
