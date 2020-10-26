#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:29:10 2020

@author: lukas
"""

import numpy as np
import csv
from ising_canonical_simulator_defs import get_energy_range
from ising_canonical_simulator_defs import evolve_lattice_unwrapped
from itertools import repeat
import os
import multiprocessing


def chunks(arr, n):
    """
    Yield successive n-sized chunks from arr.
    """
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def partial_bayesian_arr(filename, beta_arr, n, samples):
    '''
    Compiles parallel processed temp arrays into final bayesian array
    ''' 
    c = 0
    for i in range(len(beta_arr)):
        energy_range, p_dist = evolve_lattice_unwrapped(beta_arr[i], n, samples)
        if c == 0:
            with open(filename, mode='w') as file:
                csv_writer = csv.writer(file, delimiter='\t')
                csv_writer.writerow(p_dist)
                c = 1
        else:
            with open(filename, mode='a') as file:
                csv_writer = csv.writer(file, delimiter='\t')
                csv_writer.writerow(p_dist)


temp_array_path = 'PATH/TO/TEMP/ARRAY'  # directory where intermediate arrays are to be stored
final_array_path = 'PATH/TO/FINAL/ARRAY'  # directory where final arrays are to be stored

os.chdir(temp_array_path)

partitions = 50  # number of processors to use in parallel
n = 10  # n x x site size
samples = 500000000/n**2 # number of samples (make sure to keep /n**2)
samples = int(samples)


pool = multiprocessing.Pool(processes = partitions)
beta_arr = np.linspace(0.001, 5, 500)
beta_arr_partitioned = list(chunks(beta_arr, int(len(beta_arr)/partitions)))
d_filename = []
for x in range(0, partitions):
    d_filename = np.append(d_filename, "bayesian_array_n=" + str(n).zfill(2) + '_temp_' + str(x) + '.txt')
e = get_energy_range(n)

iterable = zip(d_filename, beta_arr_partitioned, repeat(n), repeat(samples))

results = pool.starmap(partial_bayesian_arr, iterable)

filename_final = "bayesian_array_n=" + str(n).zfill(2) + '_final' + '.txt'
c = 0

for filename in d_filename:
    os.chdir(temp_array_path)
    with open(filename, mode='r') as file:
        data = csv.reader(file, delimiter = '\t')
        os.chdir(final_array_path)
        for row in data:
            if c == 0:
                with open(filename_final, mode = 'w') as file:
                    csv_writer = csv.writer(file, delimiter = '\t')
                    csv_writer.writerow(row)
                    c = 1
            else:
                with open(filename_final, mode = 'a') as file:
                    csv_writer = csv.writer(file, delimiter = '\t')
                    csv_writer.writerow(row)
