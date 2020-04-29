# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:53:23 2020

@author: Lukas Herron
"""

import numpy as np
import matplotlib.pyplot as plt
import csv_funcs_local as csv
from ising_microcanonical_local import hamiltonian


def find_partition(beta, energy_dist):
    Z = 0
    for i in energy_dist:
        Z += np.exp(-beta*i)
    return Z

def find_p(beta, energy_dist):
    p_i = []
    Z = find_partition(beta, energy_dist)
    for i in energy_dist:
        p_i = np.append(p_i, np.exp(-beta*i)/Z)
    return p_i

def find_avg_energy_exp(counts, energy_dist):
    E = 0
    s = np.sum(counts)
    for i in range(len(energy_dist)):
        E += energy_dist[i]*counts[i]/s
    return E

def match_avg_energy(energies_exp, p_exp, energy_dist, energy_counts, betas):
    min_diff = 100
    min_beta = 0
    E_avg_exp = np.sum(energies_exp*p_exp)
    for i in betas:
        diff = 0
        counts_fit = energy_counts*find_p(i, energy_dist)
        p_fit = counts_fit/np.sum(counts_fit)
        E_avg_fit =  np.sum(p_fit*energy_dist)
        diff = abs(E_avg_fit - E_avg_exp)
        if diff <= min_diff:
            min_beta = i
            min_diff = diff
    return min_beta, min_diff
    
    
def get_energy_dist(energy):
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
    
def fit_beta(energy_dist, energy_counts, p_exp, betas):
    min_diff = 100
    min_beta = 0
    for i in betas:
        diff = 0
        x = energy_counts*find_p(i, energy_dist)
        p_x = x/np.sum(x)
        for j in range(len(p_x)):
            diff += abs(p_x[j] - p_exp[j])
        if diff <= min_diff:
            min_beta = i
            min_diff = diff
    return min_beta, min_diff


def small_site_dist(n):
    arr1 = []
    lat = []
    energies = []
    c = 0
    for i in range(2**n**2-1):
        s = str("{0:b}".format(i)).zfill(9)
        for j in range(len(s)):
            lat.append(s[j])
        lat = np.reshape(np.array(lat), (n,n))
        arr1.append(hamiltonian(lat))
        c += 1
        lat = []

    for i in arr1:
        c = 0
        for j in energies:
            if i == j:
                c = 1
        if c == 0:
            energies = np.append(energies, i)
    energies = np.sort(energies)
    hist, edges = np.histogram(arr1, 100)
    arr2 = []
    for i in hist:
        if i != 0:
            arr2.append(i)
    p = arr2/np.sum(arr2)
    print(p)
    print(energies)
    return energies , p, arr2
    
    
data_dir =  'temp=5_lattice=10x10_subsites=3_samples=10000_num=0079'
data_path = '/home/lukas/Ising Research/Data/local_data/' + data_dir

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
        
betas = np.linspace(0.01,20,10000)
energies_exp, p_exp = get_energy_dist(sample_energy)
energies_ss, p_ss, energy_counts = small_site_dist(3)
beta_1, diff_1 = fit_beta(energies_exp, energy_counts, p_exp, betas)
beta_2, diff_2 = match_avg_energy(energies_exp, p_exp, energies_exp, energy_counts, betas)

x_1 = energy_counts*find_p(beta_1, energies_exp)
p_1 = x_1/np.sum(x_1)

x_2 = energy_counts*find_p(beta_2, energies_exp)
p_2 = x_2/np.sum(x_2)

#%%
fig1 = plt.figure()
fig1 = plt.hist(sample_energy,40)
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Energy')

fig2 = plt.figure()
fig2 = plt.hist(sample_energy, 40)
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Sample Energy')


#%%


fig1 = plt.figure()
fig1 = plt.plot(energies_exp, p_1, '.', 'Minimized diff')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 1 Energy')

fig1 = plt.figure()
fig1 = plt.plot(energies_exp, p_2, '.', 'Matching r$<E>$')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 2 Energy')

fig1 = plt.figure()
fig1 = plt.plot(energies_exp, p_exp, '.', 'Experimental' )
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 2 Energy')

fig1 = plt.figure()
plt.plot(energies_exp, p_1, '.', 'Minimized diff')
plt.plot(energies_exp, p_2, '.', 'Matching r$<E>$')
plt.plot(energies_exp, p_exp, '.', 'Experimental' )
plt.legend()
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 1 Energy')

fig1 = plt.figure()
fig1 = plt.plot(energies_exp, p_1 - p_exp, '.')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 1 Energy Difference')

fig1 = plt.figure()
fig1 = plt.plot(energies_exp, p_2 - p_exp, '.')
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Fit 2 Energy Difference')
