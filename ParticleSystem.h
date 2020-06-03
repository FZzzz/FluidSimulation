#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

#include <GL/glew.h>
#include <vector>
#include <memory>
#include "common.h"
#include "Particle.h"

class ParticleSystem
{
public:

	ParticleSystem();
	~ParticleSystem();

	void Update();
	void AllocateParticles(size_t n);

	void setParticles(std::vector<std::shared_ptr<Particle>> particles);

	inline std::vector<std::shared_ptr<Particle>>& getParticles() { return m_particles; };
	inline GLuint getVAO() { return m_vao; };
	inline GLuint getVBO() { return m_vbo; };
	inline GLuint getEBO() { return m_ebo; };

private:

	//inner function
	void SetupParticleGLBuffers();

	void UpdateParticleGLBUfferData();

	std::vector<std::shared_ptr<Particle>> m_particles;
	std::vector<unsigned int> m_particle_indices;
	
	GLfloat m_point_sprite_size;

	GLuint m_vao;
	GLuint m_vbo;
	GLuint m_ebo;
};

#endif
