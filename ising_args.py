#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:16:42 2020

@author: lukas
"""
import numpy as np
import matplotlib.pyplot as plt
import ising_microcanonical_defs as im
import csv

# CHANGE LOC TO "HPG" FOR HIPERGATOR SIMS
# CHANGE LOC TO "local" FOR LOCAL SIMS

LOC = "local"

if LOC == "local":
    data_dir = '/home/lukas/Ising Research/Data/local_data/'
    # Directory where data is to be read and saved to
    current_dir = '/home/lukas/Ising Research/'
    # Directory where scripts are located
if LOC == "HPG":
     data_dir = '/ufrc/pdixit/lukasherron/ising_data/'
     current_dir = '/ufrc/pdixit/lukasherron/ising_sim/'
    
# INPUT PARAMETERS
#-----------------------------------------------------------------------------
subsites = 3
N = 20
m = 100
s = 1000000
n = int(s/m)
# MUST INCLUDE AS 
initial_dtype = 'temp'
mode = "load"
initial_mag = 0.8
initial_temp = 2.2
filename = "/home/lukas/Ising Research/Data/initial_lattices/20x20_initialized_lattice_temp=02.2_randint=586.txt"
#-----------------------------------------------------------------------------

if initial_dtype == 'mag':
    if mode != "load":
        lattice = im.initialize_lattice_mag(N, initial_mag)
        initial_value = initial_mag
    else:
         with open(filename, mode = 'r') as file:
             lattice = csv.reader(file, delimiter='\t')
if initial_dtype == 'temp':
     if mode != "load":
         lattice = im.initialize_lattice_temp(N, 100, initial_temp)
         initial_value = initial_temp
     else:
         with open(filename, mode = 'r') as file:
             csv_lattice = list(csv.reader(file, delimiter='\t'))
             N = (len(csv_lattice[0][:]))
             lattice = np.zeros((N,N))
             for i in range(N):
                 for j in range(N):
                     lattice[i][j] = csv_lattice[i][j]
         i = filename.find('temp')
         initial_value = float(filename[i+5:i+9])

         

time = np.linspace(0,s,int(s/m))
demon_energy, upper_bound = 16, 32

lattice, energy, mag, de, sample_energy, sample_mag, s = \
            im.main(initial_dtype, initial_value, s, N, n, lattice, demon_energy,\
                    upper_bound, m, subsites, data_dir, current_dir, 'write')

