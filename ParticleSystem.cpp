#include "ParticleSystem.h"

ParticleSystem::ParticleSystem() :
	m_point_sprite_size(5.f),
	m_vao(-1),
	m_vbo(-1),
	m_ebo(-1)
{
}

ParticleSystem::~ParticleSystem()
{
	m_particles.clear();
	m_particle_indices.clear();
}

void ParticleSystem::Update()
{
	UpdateParticleGLBUfferData();
}

void ParticleSystem::AllocateParticles(size_t n)
{
}

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

void ParticleSystem::SetupParticleGLBuffers()
{
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);
	glGenBuffers(1, &m_ebo);
	glBindVertexArray(m_vao);

	UpdateParticleGLBUfferData();
}

void ParticleSystem::UpdateParticleGLBUfferData()
{
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
	/*glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_particles.size(),
		m_particles.data(), GL_DYNAMIC_DRAW);
	*/
	glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)offsetof(Particle, m_position));
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	if (m_particle_indices.size() <= 0)
		return;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_particle_indices.size(),
		m_particle_indices.data(), GL_DYNAMIC_DRAW);

	glBindVertexArray(0);

}
