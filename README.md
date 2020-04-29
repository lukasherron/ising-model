# ising-model
Simulation and analysis of the ising model outside the Boltzmann regime.

File Index

  - 
      - Simulator for canonical ising model using Metropolis - Hastings algorithm. The simulator is primarily used in initalizing the microcanonical lattice at a specific temperature.
    
  - ising_microcanonical.py
    - Simulator for microcanonical ising model at constant energy on HPG. Simulator that evolves the lattice over time and collects energy and magnetization data of small sites and the lattice.
  
  - ising_microcanonical_local.py
    - Identical to ising_microcanonical.py except configured to run locally (i.e. data paths). Because this simulator is run locally the length of the simulation is limited.
  
  - csv_funcs.py
    - Collection of functions handling writing data to text files on HPG. Includes documentation of data formatting.
  
  - csv_funcs_local.py
    - Collection of functions handling writing data to text files on local machine, to be used with ising_microcanonical_local.py. Includes documentation of data formatting and configuring for use on local machine.
    
  - data_analysis_funcs.py
    - Collection of functions relating to data generated by ising_microcanonical.py and ising_microcanonical_local.py.
    - Functions relating to fitting Boltzmann microcanonical distribution to simulated data.
  
