#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:01:43 2020

@author: lukas
"""
import numpy as np
import matplotlib.pyplot as plt
import csv_funcs as csv_funcs
import openpyxl as op
import os
import csv

from ising_microcanonical_defs import hamiltonian
from ising_canonical_simulator_defs import match_canonical
from ising_canonical_simulator_defs import get_p_dist
from ising_canonical_simulator_defs import KL_divergence
from ising_canonical_simulator_defs import force_match
from ising_canonical_simulator_defs import evolve_lattice
from ising_canonical_simulator_defs import get_energy_range


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
    energy_range = get_energy_range(n)

    temp_num = energy_range[-1]
    energy_bins = energy_range - 0.5
    energy_bins = np.append(energy_bins, temp_num + 0.5)
    hist, edges = np.histogram(E, energy_bins)
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
    lat = np.ones((n,n))
    E = np.append(E, hamiltonian(lat))
                  
    return E

def iter_loadtxt(filename, delimiter='\t', skiprows=0, dtype=float):
    " faster way to load data"
    
    def iter_func(col):
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                yield dtype(line[col])
        iter_loadtxt.rowlength = len(line)

    data_1 = np.fromiter(iter_func(1), dtype=dtype)
    data_2 = np.fromiter(iter_func(2), dtype=dtype)
    #data = data.reshape((-1, iter_loadtxt.rowlength))
    return data_1, data_2
#%%
path_arr = []
path = '/media/lukas/RESEARCH BACKUP/sorted_simulation_data/sample_size=05/'
c = 0

for root, dirs, files in os.walk(path):
    for name in dirs:
        # READING FILENAME STRING TO FIND LATTICE SIZE AND TEMPERATURE
        print(name)
        i = name.find('subsites')
        n = int(name[i + 9:i + 10])
        print(n)
        i = name.find('temp')
        temp = float(name[i+5:i+9])

        energy_range = get_energy_range(n)
        energy_hist = np.zeros((len(energy_range)))
        
        temp_num = energy_range[-1]
        energy_bins = energy_range - 0.5
        energy_bins = np.append(energy_bins, temp_num + 0.5)
        
        #CALCULATION OF PEARSON CORRELATION
        d = 0
        sum_x = 0
        sum_y = 0
        sum_prod = 0
        sum_x_sq = 0
        sum_y_sq = 0

        new_path = path + '/' + name
        print(new_path)
        for root, dirs, files in os.walk(new_path):
             for directory in dirs:
                 print(new_path + '/' + directory)
                 for filename in os.listdir(new_path + '/' + directory):
                    
                    sample_energy_arr, boundary_energy = iter_loadtxt(new_path + '/' + directory + '/' + filename)
                    hist, edges = np.histogram(sample_energy_arr, energy_bins)
                    energy_hist += hist
                    for i in range(len(sample_energy_arr)):
                        sum_x += sample_energy_arr[i]
                        sum_y += boundary_energy[i]
                        d += 1
                        
        for filename in os.listdir(new_path):
            if filename[-4:-1] == '.tx':

                sample_energy_arr, boundary_energy = iter_loadtxt(new_path + '/' + filename)
                hist, edges = np.histogram(sample_energy_arr, energy_bins)
                energy_hist += hist
                for i in range(len(sample_energy_arr)):
                    sum_x += sample_energy_arr[i]
                    sum_y += boundary_energy[i]
                    d += 1
                    
                                
        mu_x = sum_x/d
        mu_y = sum_y/d
        
        for root, dirs, files in os.walk(new_path):
            for directory in dirs:
                print(new_path + '/' + directory)
                for filename in os.listdir(new_path + '/' + directory):
                    sample_energy_arr, boundary_energy = iter_loadtxt(new_path + '/' + directory + '/' + filename)
                        
                    for i in range(len(sample_energy_arr)):
                        sum_prod += (sample_energy_arr[i] - mu_x)*(boundary_energy[i] - mu_y)
                        sum_x_sq += (sample_energy_arr[i] - mu_x)**2
                        sum_y_sq += (boundary_energy[i] - mu_y)**2
                        
        for filename in os.listdir(new_path):
            if filename[-4:-1] == '.tx':
                sample_energy_arr, boundary_energy = iter_loadtxt(new_path + '/' + filename)
                for i in range(len(sample_energy_arr)):
                        sum_prod += (sample_energy_arr[i] - mu_x)*(boundary_energy[i] - mu_y)
                        sum_x_sq += (sample_energy_arr[i] - mu_x)**2
                        sum_y_sq += (boundary_energy[i] - mu_y)**2
                            
                        
        print("sum_x = " + str(sum_x))
        print("sum_y = " + str(sum_y))
        print("sum_x_sq = " + str(sum_x_sq))
        print("sum_y_sq = " + str(sum_y_sq))
        print("sum_prod = " + str(sum_prod))
        print("d = " + str(d))
        
        pearson_correlation = sum_prod/np.sqrt(sum_x_sq)/np.sqrt(sum_y_sq)

        if n <= 4:
        #EXPLICITLY MATCH ENERGY
            betas = np.linspace(0.0001,1,1000)
            energy_ss, p_ss, energy_counts_ss = small_site_dist(n)
            p_sim = energy_hist/np.sum(energy_hist)
            energy_range, p_sim, energy_ss, p_ss = force_match(energy_ss, p_ss, energy_range, p_sim, n) 
            p, beta = match_avg_energy(energy_range, p_sim, energy_range, p_ss, betas)
            
        #CHANGING DIRECTORIES
            cwd = os.getcwd()
            os.chdir('/home/lukas/Ising Research/Data/HiPerGator_data/plots_and_figures/plots_and_figs_' + str(n).zfill(2))
            
        #FIGURE PARAMS
            fig, ax = plt.subplots()
            ax.plot(energy_range, np.log10(p), '.', label =r'Matching $<E>$')
            ax.plot(energy_range, np.log10(p_sim), '.', label = r'Simulated')
            ax.tick_params(direction='in', length=8, width=2, top=True, right=True)
            ax.set_xlabel(r'Energy',  size='x-large')
            ax.set_ylabel(r'Frequency (log)', size='x-large')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
            for tick in ax.xaxis.get_ticklabels():
                tick.set_fontsize('x-large')
                tick.set_fontname('serif')
            for tick in ax.yaxis.get_ticklabels():
                tick.set_fontsize('x-large')
                tick.set_fontname('serif') 
            plt.title(str(n) + "x" + str(n) +' Site Energy Distribution (T = ' +str(temp) + ')',fontsize='x-large')
            plt.legend(loc='lower left')

            plt.savefig(str(n) + "x" + str(n)+"_site_"+ "T=" + str(temp)+"_energy_dist.jpg",dpi=1600)
            print('saved plot')
            os.chdir('/home/lukas/Ising Research/Data/HiPerGator_data/plots_and_figures')
            
        #COMPARE VARIENCE PER SPIN SITE
            var_p_fit = np.var(p)/n**2
            var_p_sim = np.var(p_sim)/n**2
            var_p_fit = np.var(p[0:len(p_sim)])/n**2
            var_p_sim = np.var(p_sim)/n**2
            diff = var_p_sim - var_p_fit
            
            os.chdir("/home/lukas/Ising Research/Data/HiPerGator_data/plots_and_figures/" + "plots_and_figs_" + str(n).zfill(2))
            
        #SAVING DATA TO CSV FILE
            data = [n, 1/beta,temp,  var_p_fit, var_p_sim, diff, pearson_correlation]
            filename_1 = str(n) + "x" + str(n) + "_data.txt"
            header = ["site_size", "fitted_temp", "initialized_temp","variance_fit", "variance_sim", "variance_difference", "pearson_correlation"]
            if c == 0:
                with open(filename_1, mode='w') as file:
                    csv_writer = csv.writer(file, delimiter = '\t')
                    csv_writer.writerow(header)
                c += 1
            
            with open(filename_1, mode='a') as file:
                csv_writer = csv.writer(file, delimiter = '\t')
                csv_writer.writerow(data)
            
            os.chdir(cwd)
    
        # LARGER LATTICES
        else:
            
        #CANONICAL FITTING
            beta_arr = np.linspace(0.1, 1, 1000)
            p_sim = energy_hist/np.sum(energy_hist)
            min_var, min_beta = match_canonical(p_sim, energy_range, beta_arr , n, 10000)
            e, p = evolve_lattice(min_beta, n, 100000)
            
        #CHANGING DIRECTORIES
            cwd = os.getcwd()
            os.chdir('/home/lukas/Ising Research/Data/HiPerGator_data/plots_and_figures/plots_and_figs_' + str(n).zfill(2))
            
        #FIGURE PARAMS
            fig, ax = plt.subplots()
            ax.plot(e, np.log10(p), '.', label =r'Canonical Fit')
            ax.plot(energy_range, np.log10(p_sim), '.', label = r'Simulated')
            ax.tick_params(direction='in', length=8, width=2, top=True, right=True)
            ax.set_xlabel(r'Energy',  size='x-large')
            ax.set_ylabel(r'Frequency (log)', size='x-large')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
            for tick in ax.xaxis.get_ticklabels():
                tick.set_fontsize('x-large')
                tick.set_fontname('serif')
            for tick in ax.yaxis.get_ticklabels():
                tick.set_fontsize('x-large')
                tick.set_fontname('serif') 
            plt.title(str(n) + "x" + str(n) +' Site Energy Distribution (T = ' +str(temp) + ')',fontsize='x-large')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(str(n) + "x" + str(n)+"_site_"+ "T=" + str(temp)+"_energy_dist.jpg",dpi=1600)
            
            os.chdir("/home/lukas/Ising Research/Data/HiPerGator_data/plots_and_figures/" + "plots_and_figs_" + str(n).zfill(2))
            
        #COMPARE VARIENCE PER SPIN SITE
            var_p_fit = np.var(p)/n**2
            var_p_sim = np.var(p_sim)/n**2
            var_p_fit = np.var(p[0:len(p_sim)])/n**2
            var_p_sim = np.var(p_sim)/n**2
            diff = var_p_sim - var_p_fit
        
        #SAVING DATA TO CSV FILE
            data = [n, 1/beta,temp,  var_p_fit, var_p_sim, diff, pearson_correlation]
            filename_1 = str(n) + "x" + str(n) + "_data.txt"
            header = ["site_size", "fitted_temp", "initialized_temp","variance_fit", "variance_sim", "variance_difference", "pearson_correlation"]
            if c == 0:
                with open(filename_1, mode='w') as file:
                    csv_writer = csv.writer(file, delimiter = '\t')
                    csv_writer.writerow(header)
                c += 1
            
            with open(filename_1, mode='a') as file:
                csv_writer = csv.writer(file, delimiter = '\t')
                csv_writer.writerow(data)
            
            
            os.chdir(cwd)
            
            if temp == 2.8:
                break
        
