# FluidSimulation
*Postion-based Fluid Simulation*

## Introduction
This project is an example of the basic Position-ased Fluid(PBF) simulation. Comparing with traditional SPH fluid simulation, PBF has better stability over different timesteps and faster computation speed. 


The solver I used here is a Jaccobian-fasion solver, which allows us to compute position changes concurrently. The implementation uses OpenMP for neighborhood searching and SPH computation with 8 dynamical scheduled threads. The neighbor searching algorithm is based on spatial hashing table. By cutting the whole space into grids and using hashing function to get the coordinate of neighbors, it cutted off useless comparisons and futhrer increase the performance of whole searching process. 


In near future, the simulation will be computed by GPU by exloiting the architechture of shared memory and higher degree of parallelization. 

## Result 

**OpenMP**

![](https://i.imgur.com/bjFUgTD.png)

10k fluid particles, 8 threads, 2 solver iterations

20 fps in average

**CUDA**

![](https://imgur.com/JtqN0VY.gif) 

10k fluid particles,

90 fps in average


## Analyzation by Intel VTune Profiler
![](https://i.imgur.com/WNTyFkv.png)

Effective CPU Utilization: 62.5% (5.002 out of 8 logical CPUs)

CPU Spin Time: 29.864(s) / 139.155(s)



## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

