# Ising Model Simulator
"""
Created on Mon Feb 10 16:05:19 2020

@author: Lukas Herron
"""
import numpy as np
import matplotlib.pyplot as plt

def makelattice(N):
    lattice = np.zeros((N,N),dtype=np.int8)
    for i in range(N):
        for j in range(N):
            x = int(2*np.random.random())
            if x == 1:
                lattice[i][j] = 1
            else:
                lattice[i][j] = -1
    return lattice


    
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
    J = 2
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

def heat_capacity(energy, temeprature):
    C = abs((energy[-1] - energy[-2])/(temeprature[-1] - temeprature[-2]))
    return C
#%%

N = 50
m = 10000

#%%
lattice1 = makelattice(N)
initial_lattice = lattice1
energy, mag, time, C, temperature = [], [], [], [], np.logspace(1,-1,m)
reset_counter, timer = 0, 0
t = N**2
fig6 = plt.figure()
fig6 = plt.imshow(lattice1)

for i in temperature:
    for j in range(N):
        for k in range(N):
            lattice1 = main(lattice1, j, k, i)
            timer += 1
            reset_counter += 1
            if timer == t:
                time = np.append(time,reset_counter)
                energy = np.append(energy, hamiltonian(lattice1))
                mag    = np.append(mag, magnetization(lattice1))
                if len(energy) > 1:
                    C = np.append(C, heat_capacity(energy, temperature))
                timer = 0

fig3 = plt.figure()
fig3 = plt.plot(temperature, energy)
plt.xscale('log')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title('Energy vs Temperature')

fig4 = plt.figure()
fig4 = plt.plot(temperature, mag)
plt.xscale('log')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Temperature')

fig5 = plt.figure()
fig5 = plt.hist(energy[2:],20)
plt.xlabel('Energy')
plt.ylabel('frequency')
plt.title('Energy')

fig8 = plt.figure()
fig8 = plt.hist(mag[2:],20)
plt.xlabel('Magnetization')
plt.ylabel('frequency')
plt.title('Magnetization')


fig5 = plt.figure()
fig5 = plt.imshow(lattice1)
#%%

