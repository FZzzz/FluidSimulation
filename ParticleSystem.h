#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

#include <GL/glew.h>
#include <vector>
#include <memory>
#include "common.h"
#include "Particle.h"
#include <cuda_gl_interop.h>

class ParticleSystem
{
public:

	ParticleSystem();
	~ParticleSystem();

	void Initialize();
	void InitializeCUDA();
	void Update();
	void Release();

	ParticleSet* AllocateParticles(size_t n, float particle_mass);

	//void setParticles(std::vector<std::shared_ptr<Particle>> particles);

	inline ParticleSet* getParticles() { return m_particles; };
//	inline std::vector<std::shared_ptr<Particle>>& getParticles() { return m_particles; };
	inline GLuint getVAO() { return m_vao; };
	inline GLuint getVBO() { return m_vbo; };
	inline GLuint getEBO() { return m_ebo; };
	inline cudaGraphicsResource** getCUDAGraphicsResource() { return &m_cuda_vbo_resource; };

	inline double& getUpdateTime() { return m_update_elased_time; };


private:
	
	//inner function
	void SetupCUDAMemory();
	void RegisterCUDAVBO();
	void GenerateParticleGLBuffers();

	void UpdateParticleGLBUfferData();

	ParticleSet* m_particles;
	//std::vector<std::shared_ptr<Particle>> m_particles;
	std::vector<unsigned int> m_particle_indices;
	
	GLfloat m_point_sprite_size;

	GLuint m_vao;
	GLuint m_vbo;
	GLuint m_ebo;
	
	GLuint m_cuda_vbo;

	struct cudaGraphicsResource* m_cuda_vbo_resource;

	double m_update_elased_time;
};

#endif
