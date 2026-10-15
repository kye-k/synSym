## Synaptic Diffusion Simulation

## Overview
synSym is a small tool used to simulate diffusion within a system defined by two three-dimensional geometry (.STL) files. In our use, we used this tool to simulate diffusion within a axonal geometry containing a smaller synaptic geometry (generated from ssTEM images, or CLSM images).
Particles are initialized within the geometries, and diffusion is simulated by independent random walks of each particle.

## Inputs
.JSON file:

```
{
"simulation_type": "icbs",
"trimesh_axon": "trimesh_axon_repaired.stl",
"trimesh_synapse": "trimesh_synapse_repaired.stl",
"num_p_total": 1000,
"num_p_synapse": null,
"num_p_axon": null,
"D_synapse": 0.6161550277583344,
"D_axon": 0.6161550277583344,
"tau_sim": 0.0005,
"steps": 10000,
"leave_prob": null,
"pON": 0.1,
"pOFF": 0.0,
"seed": null,
"savefile": "simulation_icbs//6_sp"
}
```

## Outputs

