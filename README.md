# ising-model
Simulation and analysis of the ising model outside the Boltzmann regime.

File Index

  - ising_canonical.py
      - Simulator for canonical ising model using Metropolis - Hastings algorithm. The simulator is primarily used in initalizing the microcanonical lattice at a specific temperature.
    
  - ising_microcanonical_defs.py
    - Definitions used in simulation of microcanonical ising model

  - csv_funcs.py
    - Collection of functions handling writing data to text files on HPG or local machine. Includes documentation of data formatting.
    
  - data_analysis_funcs.py
    - Collection of functions relating to data generated by ising_microcanonical.py and ising_microcanonical_local.py.
    - Functions relating to fitting Boltzmann microcanonical distribution to simulated data.
  
