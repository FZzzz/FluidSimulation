#include "NeighborSearch.h"
#include <omp.h>
/*
 * NavieSearch()
 * @return The neighbors of the particle 
 */


NeighborSearch::NeighborSearch(std::shared_ptr<ParticleSystem> particle_system):
    m_particle_system(particle_system)
{
}

NeighborSearch::~NeighborSearch()
{
}

void NeighborSearch::Initialize()
{
#ifdef _DEBUG
    assert(m_particle_system != nullptr);
#endif
    const auto& particles = m_particle_system->getParticles();
    m_search_cache.resize(particles.size(), std::vector<size_t>());
}

/*
 * Perform the most naive search (O(n^2)) (full search)
 */
void NeighborSearch::NaiveSearch(float effective_radius)
{
    if (m_search_cache.size() == 0 || effective_radius < 0)
        return;

    for (size_t i = 0; i < m_search_cache.size(); ++i)
        m_search_cache[i].clear();

    const auto& particles = m_particle_system->getParticles();
    float square_h = effective_radius * effective_radius;

    //#pragma omp parallel default(shared) // Personally I think this is useless... (cannot prevent race condition)
    {
        //#pragma omp for schedule(static)  // Using round-robin scheduling
        for (int i = 0; i < particles.size(); ++i)
        {
            for (int j = i+1; j < particles.size(); ++j)
            {
                float distance2 = glm::distance2(particles[i]->m_new_position, particles[j]->m_new_position);
                if (distance2 <= square_h)
                {
                    m_search_cache[i].push_back(static_cast<size_t>(j));
                    m_search_cache[j].push_back(static_cast<size_t>(i));
                }
            }
        }
    }
}

const std::vector<size_t>& NeighborSearch::FetchNeighbors(size_t i)
{
	return m_search_cache[i];
}
