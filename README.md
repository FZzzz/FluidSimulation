# FluidSimulation
*Postion-based Fluid Simulation*

## Introduction
This project implemented Position-ased Fluid(PBF) simulation with referencing to Unified Particle Physics for Real-time Applications published by Macklin and his fellow. Comparing with traditional particle-based fluid simulation, PBF has better stability over different timesteps and faster computation speed. 

## Result 

**OpenMP**

![](https://i.imgur.com/bjFUgTD.png)

10k fluid particles, 8 threads, 2 solver iterations

20 fps in average

**CUDA**

![](https://i.imgur.com/OvCaG1t.png)


* 64000 fluid particles
* 70 fps in average 


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

