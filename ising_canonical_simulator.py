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
                lattice[i][j] = 1
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

def step(lattice, i, j, T):
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

def main(steps, lattice, temp, m):
    energy, mag = [], []
    timer = 0
    for i in range(steps):
        x = int(len(lattice[0][:])*np.random.random())
        y = int(len(lattice[:][0])*np.random.random())
        lattice = step(lattice, x, y, temp[i])
        timer += 1
        if timer % m == 0:
                energy = np.append(energy, hamiltonian(lattice))
                mag    = np.append(mag, magnetization(lattice))
            

    return lattice, energy, mag


def magnetization(lattice):
    mag = np.sum(lattice)/len(lattice)**2
    return mag

def heat_capacity(energy, temeprature):
    C = abs((energy[-1] - energy[-2])/(temperature[-1] - temeprature[-2]))
    return C
#%%

N = 25
m = 10000
s = 10000000
T0 = 0.1
Tf = 4
#%%
lattice = makelattice(N)
fig6 = plt.figure()
fig6 = plt.imshow(lattice)

setting_temp = np.ones(s)*T0
lattice, en, mg = main(s, lattice, setting_temp, m)
fig6 = plt.figure()
fig6 = plt.imshow(lattice)

temp = np.linspace(T0,Tf,s)
plotting_temp = np.linspace(T0,Tf,int(s/m))

lattice, energy, mag = main(s, lattice, temp, m)

#sos = sci.butter(2, .1,  output='sos')
#energy = sci.sosfilt(sos, energy)
#mag = sci.sosfilt(sos, mag)

mag_filt = sci.savgol_filter(mag, 11, 1)



fig3 = plt.figure()
fig3 = plt.plot(plotting_temp, energy)
plt.xscale('linear')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title('Energy vs Temperature')

fig4 = plt.figure()
fig4 = plt.plot(plotting_temp, mag)
plt.xscale('linear')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Temperature')

fig4 = plt.figure()
fig4 = plt.plot(plotting_temp, mag_filt)
plt.xscale('linear')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.title('Smoothed Magnetization vs Temperature')


fig5 = plt.figure()
fig5 = plt.imshow(lattice)
#%%

