#ifndef _NEIGHBOR_SEARCH_H_
#define _NEIGHBOR_SEARCH_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <cstdint>
#include "common.h"
#include "ParticleSystem.h"

struct HashEntry
{
	std::vector<size_t> particles;
};

class NeighborSearch
{
public:
	NeighborSearch(std::shared_ptr<ParticleSystem> particle_system);
	~NeighborSearch();

	void Initialize();
	void NaiveSearch(float effective_radius);
	void SpatialSearch(float effective_radius);

	const std::vector<size_t>& FetchNeighbors(size_t i);

private:

	// Flooring function
	glm::i32vec3 Flooring(const glm::vec3& position);
	// Hashing function
	uint32_t GetHashValue(const glm::i32vec3& key);

	std::shared_ptr<ParticleSystem>	 m_particle_system;
	std::vector<std::vector<size_t>> m_search_cache;

	glm::vec3 m_grid_spacing;
	std::unordered_map<unsigned int, HashEntry*> m_hashtable;

};

#endif
