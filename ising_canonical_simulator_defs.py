# Ising Model Simulator
"""
Created on Mon Feb 10 16:05:19 2020

@author: Lukas Herron
"""
import numpy as np
from ising_microcanonical_defs import hamiltonian

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

def main(N, time, T, mode, s):
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
        if mode == 'write' and counter ==  time % s:
            energy_arr = np. append(energy_arr, hamiltonian(lattice))
    return lattice, energy_arr

