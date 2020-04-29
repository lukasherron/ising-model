# Ising Model Simulator
"""
Created on Mon Feb 10 16:05:19 2020

@author: Lukas Herron
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci

def makelattice(N):
    lattice = np.zeros((N,N),dtype=np.int8)
    for i in range(N):
        for j in range(N):
            x = int(2*np.random.random())
            if x == 1:
                lattice[i][j] = 1
            else:
                lattice[i][j] = -1
    return lattice, lattice


    
def hamiltonian(lattice):
    J = 1
    H = 0
    for i in range(N):
        for j in range(N):
            if lattice[i][j] == lattice[(i-1)% N][j]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[(i+1)% N][j]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[i][(j+1)% N]:
                H += -J/2
            else:
                H += J/2
            if lattice[i][j] == lattice[i][(j-1)% N]:
                H += -J/2
            else:
                H += J/2
    return H

def main(lattice, i, j, T):
    beta = 1/T
    J = 1
    E_0 = 0
    E_a = 0
    if lattice[i][j] == lattice[(i-1)% N][j]:
        E_0 += -J
    if lattice[i][j] == lattice[(i+1)% N][j]:
        E_0 += -J
    if lattice[i][j] == lattice[i][(j+1)% N]:
        E_0 += -J
    if lattice[i][j] == lattice[i][(j-1)% N]:
        E_0 += -J
        
    if lattice[i][j] != lattice[(i-1)% N][j]:
        E_a += -J
    if lattice[i][j] != lattice[(i+1)% N][j]:
        E_a += -J
    if lattice[i][j] != lattice[i][(j+1)% N]:
        E_a += -J
    if lattice[i][j] != lattice[i][(j-1)% N]:
        E_a += -J
        
    if  E_a > E_0:
        tolerance = np.exp(-beta*(E_a - E_0))
        x = np.random.random()
        if x < tolerance:
            lattice[i][j] = -lattice[i][j]
            
    else:
        lattice[i][j] = -lattice[i][j]
            
    return lattice

def magnetization(lattice):
    mag = np.sum(lattice)/len(lattice)**2
    return mag

#%%

N = 10
m = 1000

#%%
#lattice1, initial_lattice = makelattice(N)
#energy, mag, time, C, time = [], [], [], [], np.linspace(0,m,m)
#autocorr = []
#timer = 0
#t = N**2
#T = 1
#fig6 = plt.figure()
#fig6 = plt.imshow(lattice1)
#for i in time:
#    for j in range(N):
#        for k in range(N):
#            lattice1 = main(lattice1, j, k, T)
#            timer += 1
#            if timer == t:
#                energy = np.append(energy, hamiltonian(lattice1))
#                mag    = np.append(mag, magnetization(lattice1))
#                timer = 0
#
#fig3 = plt.figure()
#fig3 = plt.plot(time[2:], energy[2:])
#plt.xscale('linear')
#plt.xlabel('time')
#plt.ylabel('Energy')
#plt.title('Energy')
#
#fig4 = plt.figure()
#fig4 = plt.plot(time[2:], mag[2:])
#plt.xscale('linear')
#plt.xlabel('time')
#plt.ylabel('Magnetization')
#plt.title('Magnetization')
#
#fig5 = plt.figure()
#fig5 = plt.hist(energy[2:],20)
#plt.xlabel('Energy')
#plt.ylabel('frequency')
#plt.title('Energy')
#
#fig8 = plt.figure()
#fig8 = plt.hist(mag[2:],20)
#plt.xlabel('Magnetization')
#plt.ylabel('frequency')
#plt.title('Magnetization')
#
#
#fig6 = plt.figure()
#fig6 = plt.imshow(lattice1 + initial_lattice)
#
#fig7 = plt.figure()
#fig7 = plt.imshow(lattice1)
#%%

