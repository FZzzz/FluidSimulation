#ifndef _NEIGHBOR_SEARCH_H_
#define _NEIGHBOR_SEARCH_H_

#include <vector>
#include <memory>
#include <atomic>
#include "common.h"
#include "ParticleSystem.h"

class NeighborSearch
{
public:
	NeighborSearch(std::shared_ptr<ParticleSystem> particle_system);
	~NeighborSearch();

	void Initialize();
	void NaiveSearch(float effective_radius);
	const std::vector<size_t>& FetchNeighbors(size_t i);

private:

	std::shared_ptr<ParticleSystem>	 m_particle_system;
	std::vector<std::vector<size_t>> m_search_cache;
};

#endif
