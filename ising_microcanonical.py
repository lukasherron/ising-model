#Ising Model Simulator
"""
Created on Mon Feb 10 16:05:19 2020

@author: Lukas Herron
"""
import numpy as np
import ising_canonical_simulator_fixed_temp as canonical
import csv_funcs as csv

def initialize_lattice_mag(N, mag):
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
    J = 1
    H = 0
    n = len(lattice[0][:])
    for i in range(n):
        for j in range(n):
            if lattice[i][j] == lattice[(i-1)% n][j]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[(i+1)% n][j]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[i][(j+1)% n]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[i][(j-1)% n]:
                H += -J/2
            else:
                H += J/2
    return H

def step(lattice, i, j, demon_energy, upper_bound):
    J = 2
    E1 = 0
    E0 = 0
    
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

def main(steps, lattice, demon_energy, upper_bound, m, subsites, mode):
    arr = []
    lattice_energy, lattice_mag, sample_energy, sample_mag, ssites = [], [], [], [], []
    timer, num = 0,0
    if initial_dtype == 'mag':
        initial_value = initial_mag
    if initial_dtype == 'temp':
        initial_value = initial_temp
    new_dir = csv.mkdir(initial_dtype, initial_value,  N , n, subsites)
    
    for i in range(steps):
        x = int(len(lattice[0][:])*np.random.random())
        y = int(len(lattice[:][0])*np.random.random())
        lattice, demon_energy = step(lattice, x, y, demon_energy, upper_bound)
        timer += 1
        
        if timer % m == 0 and mode == 'write':
                lattice_energy = np.append(lattice_energy, hamiltonian(lattice))
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
    sample = np.zeros((length,length),dtype=np.int8)
    x = 10
    y = 10
    for i in range(length):
        for j in range(length):
            sample[i][j] = lattice[(x + i)%N][(y + j)%N]
            
    return sample

def autocorr(x):
    xp = np.fft.ifftshift((x - np.average(x))/np.std(x))
    n = xp.shape[0]
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = np.fft.fft(xp)
    p = np.absolute(f)**2
    pi = np.fft.ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

def magnetization(lattice):
    mag = float(np.sum(lattice)*len(lattice)**(-2))
    return mag


#%%
subsites = 4
N = 50
m = 1000
s = 10000000
n = int(s/m)
initial_dtype = 'temp'
initial_mag = 0.8
initial_temp = 3

if initial_dtype == 'mag':
    lattice = initialize_lattice_mag(N, initial_mag)
if initial_dtype == 'temp':
    lattice = initialize_lattice_temp(N, 10000, initial_temp)
time = np.linspace(0,s,int(s/m))
demon_energy, upper_bound = 32, 64

lattice, energy, mag, de, sample_energy, sample_mag = main(s, lattice, demon_energy, upper_bound, m)
master = csv.format_data(energy, sample_energy, mag, sample_mag)
if initial_dtype == 'mag':
    csv.main(master, initial_dtype, initial_mag,  N , n, subsites)
if initial_dtype == 'temp':
    csv.main(master, initial_dtype, initial_temp,  N , n, subsites)

