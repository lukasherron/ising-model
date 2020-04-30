#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:16:42 2020

@author: lukas
"""
import numpy as np


LOC = "local"

if LOC == "local":
    data_dir = '/home/lukas/Ising Research/Data/local_data/'
    # Directory where data is to be read and saved to
    current_dir = '/home/lukas/Ising Research/'
    # Directory where scripts are located
if LOC == "HPG":
     data_dir = '/ufrc/pdixit/lukasherron/ising_data'
     current_dir = '/ufrc/pdixit/lukasherron/ising_sim'
     
import ising_microcanonical_defs as im

subsites = 3
N = 10
m = 100
s = 1000000
n = int(s/m)
# MUST INCLUDE AS 
initial_dtype = 'temp'
initial_mag = 0.8
initial_temp = 5

if initial_dtype == 'mag':
    lattice = im.initialize_lattice_mag(N, initial_mag)
    initial_value = initial_mag
if initial_dtype == 'temp':
    lattice = im.initialize_lattice_temp(N, 10000, initial_temp)
    initial_value = initial_temp
time = np.linspace(0,s,int(s/m))
demon_energy, upper_bound = 32, 64

lattice, energy, mag, de, sample_energy, sample_mag, s = \
            im.main(initial_dtype, initial_value, s, N, n, lattice, demon_energy, upper_bound, m, subsites, 'write')
