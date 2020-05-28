# ising-model
Simulation and analysis of the ising model outside the Boltzmann regime.

# File Index

- ising_canonical.py
      - Simulator for canonical ising model using Metropolis - Hastings algorithm. The simulator is primarily used in                 initalizing the microcanonical lattice at a specific temperature.
     
- ising_args.py
    - Contains input parameters and options for microcanonical simulation. Edit this file to change input params.
    
- ising_microcanonical_defs.py
    - Definitions used in simulation of microcanonical ising model

- csv_funcs.py
    - Collection of functions handling writing data to text files on HPG or local machine. Includes documentation of data formatting.
    
- data_analysis_funcs.py
    - Collection of functions relating to data generated by ising_microcanonical.py and ising_microcanonical_local.py.
    - Functions relating to fitting Boltzmann microcanonical distribution to simulated data.
  
# Simulation Process
1. **Initializing to a temperature**\
    First, a square lattice is initialized with each point on the lattice randomly chosen to be spin up or spin down. Initializing the microcanonical lattice to a given temperature is done through simulation under canonical conditions. Points on the lattice are randomly chosen to undergo a spin flip, and the difference between the current and flipped energy contriution from the selected spin is compared to a tolerance. The tolerance is defined by 
    <a href="https://www.codecogs.com/eqnedit.php?latex=e^{-\beta&space;\Delta&space;E}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e^{-\beta&space;\Delta&space;E}" title="e^{-\beta \Delta E}" /></a>,
    where <a href="https://www.codecogs.com/eqnedit.php?latex=\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /></a> is the inverse of the temperature initialiized to.
    
 2. **Microcanonical Simulation**\
  The microcanonical simulation is initialized through several parameters listed below
    - Subsites size
    -  Total number of spin flips
    - Sampling frequency
    - Initial demon energy
    - Demon energy upper bound <br>

    The inital lattice above can either be calculated prior to the microcanonical simulation, or can be read from a saved csv file. <br>
  
    The simulation algorithm consists of randomly choosing spins on the lattice. Each step (spin flip) of the algorithm consists of finding the energy contribution from the spin before and after the potential spin flip. In the microcanonial simulation energies are calculated such that the lattice is topologically closed. The potential spin flip is accepted or rejected by comparing the difference in energy, <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;E" title="\Delta E" /></a>,  to the demon energy. The demon energy acts as an energy bank where negative <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;E" title="\Delta E" /></a> makes a positive contribution to the demon energy and positive <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;E" title="\Delta E" /></a> makes a negative contribution to the demon energy. If the proposed spin flip would cause the demon energy to be less than 0 or greater than the uooer bound the spin flip is rejected. If the demon energy is between 0 and the upper bound the flip is accepted. <br>
    
    At intervals determined by the sampling frequency a square subsite is randomly chosen from that lattice with the lattice being topologically closed. The energy of the subsite is calculated such that the subsite is topologically open, and the interaction energy across the boundary of the subsite is calculated. <br>
    
    For each sample the sample energy, magnetization, and boundary interaction energy is recorded, along with an integer corresponding the the lattice configuration if the subsite is smaller than 4x4. This bijection is found by mapping spin up to 1 and spin down to 0. Then the lattice is treated as an array and flattened to create a binary string. This string is converted to an integer that uniquely determines a subsite configuration. Additionally, the energy of the (topologically closed) lattice and it's magnetization are recorded.
    
   3. **Data Analysis**
