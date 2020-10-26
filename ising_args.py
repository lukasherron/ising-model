#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:16:42 2020

@author: lukas
"""
import numpy as np
import ising_microcanonical_defs as im
import csv

############ INPUT PARAMS ##################

# CHANGE LOC TO "HPG" FOR HIPERGATOR (COMPUTING CLUSTER) SIMS
# CHANGE LOC TO "local" FOR LOCAL SIMS
LOC = "HPG"

if LOC == "local":
    data_dir = '/PATH/TO/LOCAL/DATA/DIRECTORY'
    # Directory where data is to be read and saved to
    current_dir = '/PATH/TO/LOCAL/SIMULATION/SCRIPT/DIRECTORY'
    # Directory where scripts are located
if LOC == "HPG":
    data_dir = '/PATH/TO/HPG/DATA/DIRECTORY'
    # Directory where data is to be read and saved to
    current_dir = '/PATH/TO/HPG/SIMULATION/SCRIPT/DIRECTORY'
    # Directory where scripts are located

subsites = 9  # size of n xn samples
N = 100  # size of microcanonical universe
m = 100  # sampling frequency
s = 100000000  # total number of spin flips
n = int(s/m)

initial_dtype = 'temp'  # how to initalize universe if done in situ. Can be 'temp' or 'mag'
mode = "load"  # mode = "load" loads universe
initial_mag = 0.4  # only applicable if mode != "load"
initial_temp = 3.0  # only applicable if mode != "load"

filename = "PATH/TO/MICROCANONICAL/UNIVERSE/FILE/" + \ "MICROCANONICAL_UNIVERSE.txt"
# Example filename: "/100x100_initialized_lattice_temp=02.8.txt"

###########################################

if initial_dtype == 'mag':
    if mode != "load":
        lattice = im.initialize_lattice_mag(N, initial_mag)
        initial_value = initial_mag
    else:
        with open(filename, mode = 'r') as file:
            lattice = csv.reader(file, delimiter='\t')
if initial_dtype == 'temp':
    if mode != "load":
        lattice = im.initialize_lattice_temp(N, 100000, initial_temp)
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
