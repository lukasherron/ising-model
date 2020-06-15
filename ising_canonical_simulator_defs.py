# Ising Model Simulator
"""
Created on Mon Feb 10 16:05:19 2020

@author: Lukas Herron
"""
import numpy as np
from ising_microcanonical_defs import hamiltonian
import csv
import os


def makelattice(N):
    '''
    

    Parameters
    ----------
    N : Length of one side of NxN lattice

    -------
    Description: Randomly assigns 1 or -1 to each of the N**2 spin sites and
    returns the simulation lattice.

    '''
    lattice = np.zeros((N,N),dtype=np.int8)
    for i in range(N):
        for j in range(N):
            x = int(2*np.random.random())
            if x == 1:
                lattice[i][j] = 1
            else:
                lattice[i][j] = -1
    return lattice

def quench(lattice, i, j, T):
    '''
    Parameters
    ----------
    lattice : Simulation lattice
    i : i-th row of lattice
    j : j-th column of lattice
    T : temperature to quench to

    -------
    Description:
        
        Using the metropolis-hastings algorithm the lattice is quenched (annealed)
        to temperature T.

    '''
    beta = 1/T
    J = 1
    E0 = 0
    E1 = 0
    N = len(lattice[0][:])
    if lattice[i][j] == lattice[(i-1)% N][j]:
        E0 += -J
        E1 += J
    else:
        E0 += J
        E1 += -J
    if lattice[i][j] == lattice[(i+1)% N][j]:
        E0 += -J
        E1 += J
    else:
        E0 += J
        E1 += -J
    if lattice[i][j] == lattice[i][(j+1)% N]:
        E0 += -J
        E1 += J
    else:
        E0 += J
        E1 += -J
    if lattice[i][j] == lattice[i][(j-1)% N]:
        E0 += -J
        E1 += J
    else:
        E0 += J
        E1 += -J
        
    if  E1 > E0:
        tolerance = np.exp(-beta*(E1 - E0))
        x = np.random.random()

        if x < tolerance:
            lattice[i][j] = -lattice[i][j]
            
    else:
        lattice[i][j] = -lattice[i][j]
            
    return lattice

def main(N, time, T, mode_1, s, mode_2):
    '''

    Parameters
    ----------
    N : Length of one side of NxN simulation lattice
    time : time parameter dictating how quickly the lattice is quenched
    T : temperature to quench to 

    Returns
    -------
    lattice : Lattice quenched to temperature T

    '''
    energy_arr = []
    counter = 0
    lattice = makelattice(N)
    for i in range(len(lattice[0][:])**2*time):
        x = int(len(lattice[0][:])*np.random.random())
        y = int(len(lattice[:][0])*np.random.random())
        lattice = quench(lattice, x, y, T)
        counter +=1 
        if mode_1 == 'write' and counter ==  time % s:
            energy_arr = np. append(energy_arr, hamiltonian(lattice))
    if mode_2 == "save_config":
        cwd = os.getcwd()
        os.chdir("/ufrc/pdixit/lukasherron/initial_lattices")
        filename = str(N) + "x" + str(N) + "_initialized_lattice_temp=" + str(T).zfill(4) + "_randint=" + str(np.random.randint(1,1000)) + ".txt"
        with open(filename, mode='w') as file:
             csv_writer = csv.writer(file, delimiter = '\t')
             csv_writer.writerows(lattice)
        os.chdir(cwd)
        
    return lattice, energy_arr
 
def get_p_dist(energy_hist):
    '''
    

    Parameters
    ----------
    energy : 1D array of energies recorded in simulation

    Returns
    -------
    energies : energy 'bins' of a histogram of energy
    p : probability ofeach energy

    '''

    arr2 = []
    for i in energy_hist:
        if i != 0:
            arr2.append(i)
    p = arr2/np.sum(arr2)
    print(p)
    return  p

def get_energy_range(n):
    '''
    

    Parameters
    ----------
    n : Length of side of nxn sample site

    Returns
    -------
    energy_dist : 1D array of possible energies for a nxn lattice according to
    the relation x_{n+1} = x_{n} + x_{n-1} + 4.

    '''
    
    
    lower_bound = -4
    counter = 2
    energy_spacing  = 8
    while counter < n:
        lower_bound -= energy_spacing
        counter += 1
        energy_spacing += 4
    
    energy_dist = []

    lower_bound_lim = lower_bound
    while lower_bound <= abs(lower_bound_lim):
        energy_dist = np.append(energy_dist, lower_bound)
        lower_bound += 2

        

    energy_dist = np.delete(energy_dist,  1)
    energy_dist = np.delete(energy_dist, -2)
    
    return energy_dist

def force_match(energy_ss, p_ss, energy_sim, p_sim, n):
    '''
    Parameters
    ----------
    energy_ss : 1D array of sorted energy of small sites
    p_ss : 1D array of probabilities corresponding to p_ss
    energy_sim : 1D array of energy distribution from simulation
    p_sim : 1D array of proababilities corresponding to energy_sim

    Description:
        When the calculated distibution of energies and the simulated distribution
        of energies do not match due to some energy states not being explored in
        the simulation, force_match(...) will expand the simulated energy distribution
        and assign a probability of zero to the states that are not explored. This
        makes all of the arrays the same length.

    '''

    energy_dist = get_energy_range(n)    
        
    big_ss = np.append(energy_dist, np.zeros((len(energy_dist))))
    big_ss = np.reshape(big_ss,(2,len(energy_dist)))
    big_sim = np.zeros((2,len(energy_dist)))

    for i in range(len(big_ss[0][:])):
        big_sim[0][i] = big_ss[0][i]
    
    for i in range(len(big_ss[0][:])):
        for j in range(len(energy_ss)):
            if big_ss[0][i] == energy_ss[j]:
                big_ss[1][i] = p_ss[j]
                
    for i in range(len(energy_sim)):
        for j in range(len(big_ss[0][:])):
            if energy_sim[i] == big_ss[0][j]:
                big_sim[1][j] = p_sim[i]
                
    energy_sim_matched = big_sim[0][:]
    p_sim_matched = big_sim[1][:]
    
    energy_ss_matched = big_ss[0][:]
    p_ss_matched = big_ss[1][:]
    
    return energy_sim_matched, p_sim_matched, energy_ss_matched, p_ss_matched

def evolve_lattice(beta, N, t):
    '''
    Parameters
    ----------
    beta : Inverse temperature to be simulated at
    N : Length of NxN lattice to be simulated
    t : Number of samples

    Returns
    -------
    energy_dist1 : 1D array of possible energies for a NxN lattice
    p_dist1 : Histogram of frequencies that each energy state in energy_dist1
        was observed in simulation

    '''
    
    lattice = makelattice(N)
    energy_arr = []
    timer = 0
    while timer < t:
        for i in range(N**2*t):
            x = int(len(lattice[0][:])*np.random.random())
            y = int(len(lattice[:][0])*np.random.random())
            lattice = quench(lattice, x, y, 1/beta)
            timer += 1
            if timer % 100 == 0:
                energy_arr = np.append(energy_arr, hamiltonian(lattice))
    
    energy_dist1, p_dist1 = get_p_dist(energy_arr)
    energy_dist1 = get_energy_range(energy_arr)
    
    return energy_dist1, p_dist1

def match_canonical(p_ref, energy_ref, beta_arr, N, t):
    '''
    

    Parameters
    ----------
    p_ref : 1D array reference probability distribution.
    energy_ref : 1D array refrence energy distribution.
    beta_arr : array of beta values to text in matching to the refrence distribution.
    N : Size of NxN simulation lattice.
    t : Number of samples to be collected.

    Returns
    -------
    min_e : Average energy closest to the average energy of the refrence distribution.
    min_beta : Beta corresponding to the closest average energy.

    '''
    diff = 1
    beta_comparison = [1e5]
    while diff > 1e-3:
        avg_e_arr = []
        for i in range(len(beta_arr)):
            e, p = evolve_lattice(beta_arr[i], N, t)
            e, p, energy_ref, p_ref = force_match(energy_ref, p_ref, e, p, N)
            avg_e_sim = np.sum(p_ref*e)
            avg_e_fit = np.sum(p*e)
            avg_e_diff = avg_e_fit - avg_e_sim
            avg_e_arr = np.append(avg_e_arr, avg_e_diff)
    
        min_e = min(avg_e_arr)
        idx, = np.where(avg_e_arr == min_e)
        idx = idx[0]
        min_beta = beta_arr[idx]
        beta_comparison = np.append(beta_comparison, min_beta)
        length = abs((beta_arr[-1] - beta_arr[0])/10)
        upper_bound = min_beta + length/2
        lower_bound = min_beta - length/2
        beta_arr = np.linspace(lower_bound, upper_bound, 100)
        beta_comparison = np.delete(beta_comparison, 0)
        
    return min_e, min_beta


def KL_divergence(p_dist, p_ref):
    '''
    Parameters
    ----------
    p_dist : 1D array of first probability distribution 
    p_ref : 1D array of second probabilit distribution

    Returns
    -------
    KL_div : Kullback - Liebler Divergence between p_dist and p_ref

    '''
    KL_div = 0
    
    for i in range(len(p_dist)):
        if p_ref[i] != 0 and p_dist[i] != 0:
            ## FIX WHY KL DIV IS NEGATIVE
            KL_div += abs(p_ref[i]*np.log(p_ref[i]/p_dist[i]))

    return KL_div




