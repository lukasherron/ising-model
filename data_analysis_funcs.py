# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:53:23 2020

@author: Lukas Herron
"""

import numpy as np
import matplotlib.pyplot as plt
import csv_funcs as csv
from ising_microcanonical_defs import hamiltonian


def find_partition(beta, energy_dist):
    ''' NOT IN USE. NEED TO FIX'''
    Z = 0
    for i in energy_dist:
        Z += np.exp(-beta*i)
    return Z

def find_p(beta, energy_dist):
    ''' NOT IN USE. NEED TO FIX'''
    p_i = []
    for i in energy_dist:
        p_i = np.append(p_i, np.exp(-beta*i))
    
    return p_i/np.sum(p_i)

def match_avg_energy(energy_sim, p_sim, energy_ss, p_ss, betas):
    '''
    

    Parameters
    ----------
    energy_sim :1D array of simulation energy distribution
    p_sim : 1D array of probabilities corresponding to energy_sim
    energy_ss : 1D array of small site energy distribution
    p_ss : 1D array of probabilities corresponding to energy_ss
    betas : array of beta values to match

    Returns
    -------
    p_fit : probability distribution corresponding to matched <E>
    beta_fit : beta corresponding to matched <E> distribution

    '''
    p_i = []
    avg_E_sim = np.sum(p_sim*energy_sim)
    diff_fit = 100
    beta_fit = 0
    for beta in betas:
        for i in energy_sim:
            p_i = np.append(p_i, np.exp(-beta*i))
            p_i = p_i/np.sum(p_i)
        p_p = p_ss*p_i
        p_p = p_p/np.sum(p_p)
        p_p = p_p[::-1]
        avg_E_fit = np.sum(p_p*energy_ss)
        if abs(avg_E_sim - avg_E_fit) < diff_fit:
            diff_fit = abs(avg_E_sim - avg_E_fit)
            beta_fit = beta
            p_fit = p_p
        p_i = []
    return p_fit, beta_fit


def fit_beta(energy_dist, energy_counts, p_exp, betas):
    ''' NEED TO FIX '''
    min_diff = 100
    min_beta = 0
    for beta in betas:
        diff = 0
        x = energy_counts*find_p(beta, energy_dist)
        p_x = x/np.sum(x)
        for j in range(len(p_x)):
            diff += abs(p_x[j] - p_exp[j])
        if diff <= min_diff:
            min_beta = beta
            min_diff = diff
    return min_beta, min_diff
    
    
def get_energy_dist(energy):
    '''
    

    Parameters
    ----------
    energy : 1D array of energies recorded in simulation

    Returns
    -------
    energies : energy 'bins' of a histogram of energy
    p : probability ofeach energy

    '''
    energies = []
    for i in energy:
        c = 0
        for j in energies:
            if i == j:
                c = 1
        if c == 0:
            energies = np.append(energies, i)
    energies = np.sort(energies)
    hist, edges = np.histogram(energy, 100)
    arr2 = []
    for i in hist:
        if i != 0:
            arr2.append(i)
    p = arr2/np.sum(arr2)
    print(p)
    print(energies)
    return energies , p
    


def small_site_dist(n):
    '''
    

    Parameters
    ----------
    n : size of lattice to calculate energy distribution of
   
    -------
    
    Description:
        Returns the distribution of all possible energies of an nxn lattice
    
    Returns
    -------
    energies : array of possible energy values
    p : array of probabilites of each energy value
    counts : counts of each energy value

    '''
    energies = []
    E = binary2energy(n)
    
    #Eliminating repeat energies and recording counts
    for i in E:
        c = 0
        for j in energies:
            if i == j:
                c = 1
        if c == 0:
            energies = np.append(energies, i)

    energies = np.sort(energies)
    hist, edges = np.histogram(E, 100)
    counts = []
    for i in hist:
        if i != 0:
            counts.append(i)
    p = counts/np.sum(counts)
    
    return energies , p, counts


def binary2energy(n):
    '''

    Parameters
    ----------
    n : size of lattice to calculate energy distribution of

    -------
    Description:
        By expressing the lattice configuration as binary numbers the energies
        of all the possible configurations of an nxn lattice are found.
    
    Returns
    -------
    E: Energy array

    '''
    E = []
    lat = np.array([])
    for i in range(2**n**2-1):
        s = str("{0:b}".format(i)).zfill(n**2)
        for j in range(len(s)):
            if s[j] == str(1):
                lat = np.append(lat, 1)
            else:
                lat = np.append(lat, -1)
        lat = np.reshape(lat, (n,n))
        E = np.append(E, hamiltonian(lat))
        lat = []
        E = np.append(E, hamiltonian(np.ones((n,n))))
    return E

def force_match(energy_ss, p_ss, energy_sim, p_sim):
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
    big_ss = np.append(energy_ss, p_ss)
    big_ss = np.reshape(big_ss,(2,len(energy_ss)))
    big_sim = np.zeros((2,len(energy_ss)))
    # print(big_sim)
    print(big_ss)
    for i in range(len(big_ss[0][:])):
        big_sim[0][i] = big_ss[0][i]
        for k in energy_sim:
            for j in range(len(big_ss[0][:])):
                if k == big_ss[0][j]:
                    big_sim[1][j] = p_sim[j]
    print(big_sim)
    energy_sim_matched = big_sim[0][:]
    p_sim_matched = big_sim[1][:]
    return energy_sim_matched, p_sim_matched
                
            

#%%
    
data_path = '/home/lukas/Ising Research/Data/local_data/temp=2.2_lattice=20x20_subsites=4_samples=10000_num=0195'

energy =        csv.loaddata(data_path, 0)
sample_energy = csv.loaddata(data_path, 1)
mag =           csv.loaddata(data_path, 2)
sample_mag =    csv.loaddata(data_path, 3)

hist, edges = np.histogram(energy, 100)
counts = []
for i in hist:
    if i != 0:
        counts.append(i)
        
#%%
        
betas = np.linspace(0.001,10,1000)

#%%
energy_ss, p_ss, energy_counts_ss = small_site_dist(4)

#%%


energy_sim, p_sim = get_energy_dist(sample_energy)
energy_sim, p_sim = force_match(energy_ss, p_ss, energy_sim, p_sim)
beta_1, diff_1 = fit_beta(energy_sim, energy_counts_ss, p_sim, betas)
p_2, beta_2 = match_avg_energy(energy_sim, p_sim, energy_ss, p_ss, betas)

x_1 = p_ss*find_p(beta_1, energy_sim)
p_1 = x_1/np.sum(x_1)


#%%


print(beta_fit)
    
fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_fit, '.', label = r'Matched <E>')
fig1 = plt.plot(energy_sim, p_sim, '.', label =r'Simulated' )
plt.legend()
plt.ylabel('frequency')
plt.title('Fit 1 Energy')

fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_1, '.', label = r'Minimized diff')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 1 Energy')
        
        

#%%
# fig1 = plt.figure()
# fig1 = plt.hist(sample_energy,40)
# plt.xlabel('Energy')
# plt.ylabel('frequency')
# plt.title('Energy')

fig2 = plt.figure()
fig2 = plt.hist(sample_energy, 100)
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Sample Energy')
plt.xlim(-28,28)


#%%


fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_1, '.', label = r'Minimized diff')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 1 Energy')

fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_2, '.', label = r'Matching r$<E>$')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 2 Energy')

fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_sim, '.', label =r'Experimental' )
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Exp Energy')

fig1 = plt.figure(dpi = 1600)
#plt.plot(energy_sim, p_1, '.', label = r'Minimized diff')
plt.plot(energy_sim, p_2, '.', label =r'Matching $<E>$')
plt.plot(energy_sim, p_sim, '.', label = r'Experimental' )
plt.legend()
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit Energy')
plt.savefig('boltzmann_verification',dpi=1600)

fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_1 - p_sim, '.')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 1 Energy Difference')

fig1 = plt.figure()
fig1 = plt.plot(energy_sim, p_2 - p_sim, '.')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 2 Energy Difference')

#%%
energies_ss, p_ss, energy_counts = small_site_dist(4)
fig1 = plt.figure()
fig1 = plt.plot(energies_ss, p_ss, '.', label = r'small sites')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Small Site Energy')
plt.xlim(-28,28)
