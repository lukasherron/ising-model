# Microcanonical Ising Model Local Simulator
"""
Created on Mon Feb 10 16:05:19 2020

@author: Lukas Herron
"""
import numpy as np
import ising_canonical_simulator_fixed_temp as canonical
import csv_funcs_local as csv

def initialize_lattice_mag(N, mag):
    '''
    Parameters
    ----------
    N   : Length of one edge of the lattice so that the lattice contains N^2 spins.
    
    mag : Magnetization to initialize the lattice at. -1 < mag < 1.
    
    -------
        
    Description:
        
    Returns a NxN array (lattice) initialzed at magnetization = mag.

    Returns

    '''
    lattice = np.zeros((N,N),dtype=np.int8)
    prop = abs((mag+1)/2*N**2)
    counter = 0
    while counter < prop:
        i = int(N*np.random.random())
        j = int(N*np.random.random())
        if mag <= 0 and lattice[i][j] == 0:
            lattice[i][j] = -1
            counter += 1
        if mag > 0 and lattice[i][j] == 0:
            lattice[i][j] = 1
            counter += 1
    for x in range(N):
        for y in range(N):
            if lattice[x][y] == 0:
                if mag <= 0:
                    lattice[x][y] = 1
                if mag > 0:
                    lattice[x][y] = -1
    return lattice

def initialize_lattice_temp(N, time, T):
    '''
    Parameters
    ----------
    N       : Length of one edge of the lattice so that the lattice contains N^2 spins.
    
    time    : Length of time to perform simulated annealing to temperature = temp.
    
    T       : Temperature to initialize the lattice at.
    
    ----------
        
    Description:
        
    Returns a NxN array (lattice) initialzed at temperature = temp by simulated 
    annealing to temp. Has dependence on ising_canonical_simulator_fixed_temp.py.

    '''
    lattice = np.zeros((N,N),dtype=np.int8)
    for i in range(N):
        for j in range(N):
            x = int(2*np.random.random())
            if x == 1:
                lattice[i][j] = 1
            else:
                lattice[i][j] = -1
        
    for i in range(time):
        for j in range(N):
            for k in range(N):
                lattice = canonical.main(lattice, j, k, T)
    return lattice
            
    
def hamiltonian(lattice):
    '''
    Parameters
    ----------
    lattice : 2D array that energy is to be calculated for.
    
    ----------
        
    Description:
        
    Calculates energy of 2D array using spin-spin interaction energy J. -J/2 is
    added if neighboring spins are the same, J/2 is added if neighboring spins
    are opposite.
    
    Returns the energy of the lattice.

    Notes:
    - J/2 is added in each case to account for the double counting of 
    each spin interaction. 
    - Assumes the lattice is topologically OPEN.

    '''
    J = 1
    H = 0
    n = len(lattice[0][:])
    for i in range(n):
        for j in range(n):
            if i > 0:
                if lattice[i][j] == lattice[(i-1)% n][j]:
                    H += -J/2
                else:
                    H += J/2
            if i < n - 1:
                if lattice[i][j] == lattice[(i+1)% n][j]:
                    H += -J/2
                else:
                    H += J/2
            if j < n - 1:
                if lattice[i][j] == lattice[i][(j+1)% n]:
                    H += -J/2
                else:
                    H += J/2
            if j>0:
                if lattice[i][j] == lattice[i][(j-1)% n]:
                    H += -J/2
                else:
                    H += J/2
    return H


def hamiltonian_wrapped(lattice):
    '''
    Parameters
    ----------
    lattice : 2D array that energy is to be calculated for.
    
    ----------
        
    Description:
        
    Calculates energy of 2D array using spin-spin interaction energy J. -J/2 is
    added if neighboring spins are the same, J/2 is added if neighboring spins
    are opposite.
    
    Returns the energy of the lattice.

    Notes:
    - J/2 is added in each case to account for the double counting of 
    each spin interaction. 
    - Assumes the lattice is topologically CLOSED (i.e. hamiltonian is wrapped)

    '''
    J = 1
    H = 0
    n = len(lattice[0][:])
    for i in range(n):
        for j in range(n):
            if lattice[i][j] == lattice[(i-1)%n][j]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[(i+1)%n][j]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[i][(j+1)%n]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[i][(j-1)%n]:
                H += -J/2
            else:
                H += J/2
    return H


def step(lattice, i, j, demon_energy, upper_bound):
    '''
    Parameters
    ----------
    lattice : 2D array to be evolved over a single step.
    
    i : i^th column of the lattice (spin located at (i,j))
    
    j : j^th row of the lattice (spin located at (i,j))
    
    demon_energy : Initial energy used to keep the lattice at approx. constant
                    energy.
                    
    upper_bound : Maximum value that demon_energy is able to reach.
    -------
    
    Description:

    A step is defined as the proposed change of a single randomly selected spin
    at lattice position (i,j). The demon_energy is a fictituous quantity that
    tracks if a spin adds or takes away energy from the lattice. The proposed
    spin change is allowed so long as the result of the change is such that
    0 <= demon_energy <= upper_bound.
    
    Returns the lattice after the proposed change is accepted or rejected, and
    the demon_energy.
    
    '''
    J = 2
    E1 = 0
    E0 = 0
    
    N = len(lattice[0][:])
    
    if lattice[i][j] == lattice[(i-1)% N][j]:
        E0 += -J
    if lattice[i][j] == lattice[(i+1)% N][j]:
        E0 += -J
    if lattice[i][j] == lattice[i][(j+1)% N]:
        E0 += -J
    if lattice[i][j] == lattice[i][(j-1)% N]:
        E0 += -J
    
    if lattice[i][j] != lattice[(i-1)% N][j]:
        E1 += -J
    if lattice[i][j] != lattice[(i+1)% N][j]:
        E1 += -J
    if lattice[i][j] != lattice[i][(j+1)% N]:
        E1 += -J
    if lattice[i][j] != lattice[i][(j-1)% N]:
        E1 += -J
    
    dE = E1 - E0
    if dE <= demon_energy and demon_energy - dE <= upper_bound and demon_energy - dE >= 0:
        demon_energy -= dE
        lattice[i][j] *= -1
    
    return lattice, demon_energy

def main(initial_dtype, initial_value, steps, N, n, lattice, demon_energy, upper_bound, m, subsites, mode):
    '''
    Parameters
    ----------
    steps : The number of times the step(...) function is to be called.
    
    lattice : 2D array to be evolved of over steps = s.
    
    demon_energy : Initial energy used to keep the lattice at approx. constant
                    energy.
                    
    upper_bound :  Maximum value that demon_energy is able to reach.
    
    m : Data  is to be collected an stored every m steps.
    
    subsites : The size of the small sites to be analyzed.
                Ex : subsites = 3 means a 3x3 site will randomly selected 
                to be analyzed.
                
    mode : Set to "write" is data is to be written to .txt files.
            Set to "none" if data is not to be stored.
    -------
    
    Description:
        
    Main loop that controls the evolution of the lattice. The step(...) function 
    is called steps = s times. Every m times step(...) is called the lattice 
    energy and magnetization is calculated, and a MxM (subsites = M) small 
    site is sampled from the lattice, and its energy and magnetization is recorded.
    These data are written to .txt files in a directory created using csv_funcs.py.
    
    Returns lattice, lattice_energy, lattice_mag, arr (demon_energy), sample_energy
        sample_mag, ssites. (These are only used for debugging purposes and testing)
        
    Notes: 
        - The energy of the lattice must be calculated using hamiltonian_wrapped
            (lattice must be assumed topologically closed) for the energy to be kept
            constant.
        
        - The energy of the small sites is calculated as if they are topologically
            open. This is so the desired edge effects cause deviation from the
            canonical ensemble.
        
        - With "write" enabled, a directory is created at a location specified
            in csv_funcs.py where the data is saved according to a naming scheme also
            documented in csv_funcs.py.
        
    '''
    arr = []
    lattice_energy, lattice_mag, sample_energy, sample_mag, ssites = [], [], [], [], []
    timer, num = 0,0
    new_dir = csv.mkdir(initial_dtype, initial_value,  N , n, subsites)
    
    for i in range(steps):
        x = int(len(lattice[0][:])*np.random.random())
        y = int(len(lattice[:][0])*np.random.random())
        lattice, demon_energy = step(lattice, x, y, demon_energy, upper_bound)
        timer += 1
        
        if timer % m == 0 and mode == 'write':
                lattice_energy = np.append(lattice_energy, hamiltonian_wrapped(lattice))
                lattice_mag    = np.append(lattice_mag, "{:.3f}".format(magnetization(lattice)))
                arr = np.append(arr, demon_energy)
                sample, int_f = random_sample_sites(lattice, subsites)
                ssites = np.append(ssites, int_f)
                sample_energy = np.append(sample_energy, hamiltonian(sample))
                sample_mag = np.append(sample_mag, "{:.3f}".format(magnetization(sample)))

        if len(lattice_energy) == 10000:
            master = csv.format_data(lattice_energy, sample_energy, lattice_mag, sample_mag, ssites)
            filename = csv.create_file(initial_dtype, initial_value , N, n, subsites, num)
            csv. write_to_file(filename, master, new_dir)
            lattice_energy, lattice_mag, sample_energy, sample_mag = [], [], [], []
            num += 1
            
            
    return lattice, lattice_energy, lattice_mag, arr, sample_energy, sample_mag, ssites

def random_sample_sites(lattice, length):
    '''
    Parameters
    ----------
    lattice : 2D array which the small sites are sampled from.
    
    length : The length of one edge of the small site.
            Ex: length = N will sample small sites of size NxN.
    -------
        
    Description:
        
    Random selection of small sites to be sampled.
    
    Returns an NxN (length = N) array of the sample and an integer representation
    of the configuration of the lattice.
    
    Notes:
        - The lattice configuration is stored as an integer. Flattening the 2D sample
            array gives a string of the form 111-1-1...1. The transformation -1 --> 0
            is apllied to give a binary number, that is then converted to a base 10 int.
            Each integer corresponds to a unique configuration of the small site.

    '''
    N = len(lattice[0][:])
    sample = np.zeros((length,length),dtype=np.int8)
    x = int(((len(lattice[0][:])-length))*np.random.random())
    y = int(((len(lattice[:][0])-length))*np.random.random())
    for i in range(length):
        for j in range(length):
            sample[i][j] = lattice[(x + i)%N][(y + j)%N]
    flattened = np.ndarray.flatten(sample)
    str_f = list(flattened)
    s = ''
    for i in range(len(str_f)):
        if str_f[i] == -1:
            str_f[i] = 0
    for i in range(len(str_f)):
        s += str(str_f[i])
    int_f = int(s, 2)
    
    
            
    return sample, int_f

def constant_sample_sites(lattice, length):
    '''
    Parameters
    ----------
    lattice : 2D array which the small sites are sampled from.
    
    length : The length of one edge of the small site.
            Ex: length = N will sample small sites of size NxN.
    -------
        
    Description:
        
    Selection of a small sites to be sampled over the course of main(...).
    
    Returns an NxN (length = N) array of the sample
    
    Notes:
       - Currently not in use in main(...), but is able to replace random_sample_sites.
       - Unlike random_sample_sites the chosen small site is at the same location
           within the lattice for the entire simulation.
       - Site location parameters x,y must be changed within the definition.
       
    '''
    N = len(lattice[0][:])
    sample = np.zeros((length,length),dtype=np.int8)
    x = 10
    y = 10
    for i in range(length):
        for j in range(length):
            sample[i][j] = lattice[(x + i)%N][(y + j)%N]
            
    return sample

def autocorr(x):
    '''
    Parameters
    ----------
    x : 1D array for autocorrelation ot be performed on.

    -------
        
    Description:
        
    Calculates the autocorrelation of a 1D array using FFT.
        
    Returns a 1D array of the autocorrelation of array x.
    
    Notes:
        - Should be moved to data_analysis_funcs.py.
        - Used to determine the sampling frequency of small sites so that they
            are uncorrelated between sucessive samples.

    '''
    xp = np.fft.ifftshift((x - np.average(x))/np.std(x))
    n = xp.shape[0]
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = np.fft.fft(xp)
    p = np.absolute(f)**2
    pi = np.fft.ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

def magnetization(lattice):
    '''
    Parameters
    ----------
    lattice : 2D array to calculate magnetization of.
     
    -------
    Description:
        Calculates the magnetization of a lattice by taking the sum of the lattice
        spin values (1 or -1) and dividing the sum by the area of the lattice.
        
    Returns magnetization  = mag

    '''
    mag = np.sum(lattice)/len(lattice)**2
    return mag

def heat_capacity(energy, temeprature):
    '''
    Parameters
    ----------
    energy : TYPE
        DESCRIPTION.
    temeprature : TYPE
        DESCRIPTION.
    -------
    
    Unfinished function to be added to data analysis funcs that will calculate
    the heat capacity of the lattice as a function of energy and temperature.
    Should be moved to data_analysis_funcs.py

    '''
    C = abs((energy[-1] - energy[-2])/(temeprature[-1] - temeprature[-2]))
    return C


