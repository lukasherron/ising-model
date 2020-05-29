
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:53:23 2020

@author: Lukas Herron
"""

import numpy as np
import matplotlib.pyplot as plt
import csv_funcs as csv
from ising_microcanonical_defs import hamiltonian
from ising_canonical_simulator_defs import match_canonical
from ising_canonical_simulator_defs import get_energy_dist
from ising_canonical_simulator_defs import KL_divergence
from ising_canonical_simulator_defs import force_match



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
    p = []
    avg_E_sim = np.sum(p_sim*energy_sim)
    diff_fit = 1000
    for beta in betas:
        for i in range(len(energy_sim)):
            p = np.append(p, p_ss[i]*np.exp(-beta*energy_sim[i]))
        p = p/np.sum(p)
        avg_E_fit = np.sum(p*energy_ss)
        if abs(avg_E_sim - avg_E_fit) < diff_fit:
            diff_fit = abs(avg_E_sim - avg_E_fit)
            beta_fit = beta
            p_fit = p
        p = []
    return p_fit, beta_fit

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
    hist, edges = np.histogram(E, 1000)
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
        lat = np.array([])
    # array of all ones (spin up) are not calculated from a binary number.
    E = np.append(E, hamiltonian(np.ones((n,n))))
                  
    return E


                
            

#%%
data_path_1 = '/home/lukas/Ising Research/Data/HiPerGator_data/categorized_simulation_data/sample_size=04/temp=2.2_lattice=100x100_subsites=4_samples=100000_num=0017'
n = 4

# RETURNS DATA IN HISTOGRAM FORMAT
#energy =        csv.loaddata(data_path_1, 0)
sample_energy, energy_sim = csv.loaddata(data_path_1, 1)
#mag =           csv.loaddata(data_path_1, 2)
#sample_mag =    csv.loaddata(data_path_1, 3)

        
betas = np.linspace(0.0001,1,1000)
energy_ss, p_ss, energy_counts_ss = small_site_dist(n)
p_sim = get_energy_dist(sample_energy)
print(len(p_sim))
print(len(energy_sim))
energy_sim, p_sim, energy_ss, p_ss = force_match(energy_ss, p_ss, energy_sim, p_sim, n)

p_2, beta_2 = match_avg_energy(energy_sim, p_sim, energy_ss, energy_counts_ss, betas)

div = KL_divergence(p_sim, p_2)
print(div)

fig2 = plt.figure()
fig2 = plt.hist(sample_energy, 120)
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Sample Energy')
plt.xlim(-28,28)


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
plt.title('Energy Distribution')

fig1 = plt.figure(dpi = 1600)
#plt.plot(energy_sim, p_1, '.', label = r'Minimized diff')
plt.plot(energy_sim, np.log10(p_2), '.', label =r'Matching $<E>$')
plt.plot(energy_sim, np.log10(p_sim), '.', label = r'Experimental' )
plt.legend()
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Energy Distribution (log plot)')
#plt.savefig('boltzmann_verification',dpi=1600)

fig1 = plt.figure()
fig1 = plt.plot(energy_ss,np.log10(p_ss), '.', label = r'small sites')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Small Site Energy')
plt.xlim(-28,28)

#%%


data_path_2 = '/ufrc/pdixit/lukasherron/ising_data/temp=2.5_lattice=100x100_subsites=3_samples=100000_num=0012'
n = 4

energy_2 =        csv.loaddata(data_path_2, 0)
sample_energy_2 = csv.loaddata(data_path_2, 1)
mag_2 =           csv.loaddata(data_path_2, 2)
sample_mag_2 =    csv.loaddata(data_path_2, 3)

energy_sim_2, p_sim_2 = get_energy_dist(sample_energy_2)

beta_arr = np.linspace(0.1, 1, 100)
min_KL, min_beta = match_canonical(p_sim_2, energy_sim_2, beta_arr , n, 10000)


#%%
from ising_canonical_simulator_defs import evolve_lattice
n = 6
beta_range = np.linspace(0.1, 1, 100)
KL_arr = []
for i in range(len(beta_range)):
    e, p = evolve_lattice(beta_range[i], n, 10000)
    e, p, energy_sim_2, p_sim_2 = force_match(energy_sim_2, p_sim_2, e, p, n)
    KL = KL_divergence(p_sim_2, p)
    KL_arr = np.append(KL_arr, KL)
    

fig1 = plt.figure()
plt.plot(beta_range, KL_arr, '.')
plt.xlabel(r"$\beta$")
plt.ylabel('Kullback - Leibler Divergence')
plt.title(r"6x6 site KL divergence vs $\beta$")

#%%
N = n
var_p_fit = np.var(p_2)/N**2
var_p_sim = np.var(p_sim)/N**2

    

