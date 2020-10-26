#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:01:43 2020

@author: lukas
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math

from ising_microcanonical_defs import hamiltonian
from ising_canonical_simulator_defs import force_match
from ising_canonical_simulator_defs import get_energy_range
from scipy.stats import invgamma


def KL_divergence(p_ref, p_dist):
    '''
    Parameters
    ----------
    p_dist : 1D array of first probability distribution
    p_ref : 1D array of second probabilit distribution

    Returns
    -------
    KL_div : Kullback - Liebler Divergence between p_dist and p_ref

    '''
    KL_div = 0
    if len(p_dist) < len(p_ref):
        L = len(p_dist)
    else:
        L = len(p_ref)
    for i in range(L):
        if p_ref[i] > 1e-8 and p_dist[i] > 1e-8:
            KL_div += p_ref[i]*np.log(p_ref[i]/p_dist[i])

    return (KL_div)


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

    Returns
    -------
    energies : array of possible energy values
    p : array of probabilites of each energy value
    counts : counts of each energy value

    '''
    energies = []
    E = binary2energy(n)
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

    return energies, p, counts


def binary2energy(n):
    '''
    Parameters
    ----------
    n : size of lattice to calculate energy distribution of

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
        lat = np.reshape(lat, (n, n))
        E = np.append(E, hamiltonian(lat))
        lat = np.array([])
    # array of all ones (spin up) are not calculated from a binary number.
    lat = np.ones((n, n))
    E = np.append(E, hamiltonian(lat))

    return E


def iter_loadtxt(filename, mode, col_1, col_2, delimiter='\t', skiprows=0, dtype=float):
    " faster way to load data"

    def iter_func(col):
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                yield dtype(line[col])
                iter_loadtxt.rowlength = len(line)
    if mode == "double":
        data_1 = np.fromiter(iter_func(col_1), dtype=dtype)
        data_2 = np.fromiter(iter_func(col_2), dtype=dtype)
        return data_1, data_2
    if mode == 'single':
        data_1 = np.fromiter(iter_func(col_1), dtype=dtype)
        return data_1


def get_degeneracy(energy, energy_ss):
    '''
    Parameters
    ----------
    energy: energy values in energy range (computed by get_energy_range)
    energy_ss: energy distribution for all microstates of an nxn site (computed
    by small_site_dist)

    Returns
    ---------
    idx: array of indicies where 'energy' is degenerate
    '''
    [idx] = np.where(energy_ss == energy)
    return idx


def small_sites_get_fit(beta_arr, p_1, lam, alpha):
    '''
    Parameters
    -----------
    beta_arr: array of beta values
    p_1: bayesian array transformed to energy pdf
    lam: lambda value for beta distribution
    alpha: alpha value for beta distribution

    Returns:
    ----------
    arr: Un-normalized energy distribution
    p_1: bayesian array transformed to energy pdf
    p_beta: beta distribution
    '''

    p_beta = np.zeros((len(beta_arr), 1))
    p_beta = invgamma.pdf(beta_arr, alpha, loc=0, scale=lam)
    arr = np.matmul(p_1, p_beta)

    return arr, p_1, p_beta


def load_bayesian(n, energy_range, bayesian_array_path):
    '''
    Parameters
    -----------
    n: Size of nxn ising model in any one direction
    energy_range: array of possible energy_values for an nxn ising model, as
    calculated by get_energy_range(n)
    bayesian_array_path: /PATH/TO/BAYESIAN/ARRAYS

    Returns
    ----------
    p_1: bayesian array transformed to energy pdf
    '''

    os.chdir(bayesian_array_path + str(n).zfill(2))
    filename = "bayesian_array_n=" + str(n).zfill(2) + "_final_1.txt"
    arr = iter_loadtxt(filename, 'single', 0, 0)
    L = len(arr)
    p_1 = np.zeros((L, len(energy_range)))
    for i in range(len(energy_range)):
        filename = "bayesian_array_n=" + str(n).zfill(2) + "_final_1.txt"
        arr = iter_loadtxt(filename, 'single', i, 0)
        for j in range(L):
            p_1[j][i] = arr[j]
    print(np.shape(p_1))
    return p_1


def large_sites_get_fit(energy_arr, n, lam, alpha, beta_grating, L, beta_arr, bayesian_array_path):
    '''
    Parameters
    ----------
    energy_arr: array of possible energy_values for an nxn ising model, as
    calculated by get_energy_range(n)
    N: size of NxN ising model in any one dimension
    lam: lambda value for beta distribution
    alpha: alpha value for beta distribution
    L: length of beta array
    beta_arr: beta array
    bayesian_array_path: /PATH/TO/BAYESIAN/ARRAYS

    Returns:
    ----------
    arr: Un-normalized energy distribution
    p_1: bayesian array transformed to energy pdf
    p_beta: beta distribution
    '''

    p_1 = load_bayesian(n, energy_arr, bayesian_array_path)
    (r, c) = np.shape(p_1)
    print(np.shape(p_1))
    p_beta = np.zeros((r, 1))

    for i in range(len(energy_range)):
        filename = "bayesian_array_n=" + str(n).zfill(2) + "_final_1.txt"
        arr = iter_loadtxt(filename, 'single', i, 0)
        for j in range(L):
            p_1[j][i] = arr[j]
    for row in range(r):
        Z = 0
        for col in range(c):
            Z += p_1[row][col]
        for col in range(c):
            p_1[row][col] = p_1[row][col]/Z

    p_beta = invgamma.pdf(beta_arr, alpha, loc=0, scale=lam)
    p_1 = np.transpose(p_1)
    arr = np.matmul(p_1, p_beta)

    return arr, p_1, p_beta


def log_factorial(n):
    '''
    Parameters
    -----------
    n: integer

    Returns
    -----------
    log(n!) via Stirling's approximation
    '''

    return 0.5*np.log(2*math.pi*n) + n*np.log(n) - n


# INITIAL PARAMS
'''
In order for the data analysis script to run, the file structure must be as
follows:

Each of the below path strings should be to a directory containing the relevant
data. Within each of the directories, there must be subdirectories with naming
convention 'sample_size=01', 'sample_size=02', 'sample_size=03', etc. The
relevant data must be organized by sample size into the subdirectories, with
the naming scheme specified in csv_funcs.py.

NOTE: The path strings below refer to the directory containing the
subdirectories
'''

boundary_energy_path = 'PATH/TO/BOUNDARY/ENERGIES'
system_energy_path = 'PATH/TO/SYSTEM/ENERGIES'
joint_energy_path = 'PATH/TO/JOINT/ENERGIES'
plots_and_figs_path = 'PATH/TO/PLOTS/AND/FIGS'
bayesian_array_path = 'PATH/TO/BAYESIAN/ARRAYS'

path_arr = ['/PATH/TO/DATA']
fit_type = "KL"
load_data = "false"

for path in path_arr:
    d, c_1, c_2, c_l, c_mutual_inf = 0, 0, 0, 0, 0
    for root, dirs, files in os.walk(path):
        dirs = sorted(dirs)
        if d == 1:
            break
        for name in dirs:

            i = name.find('subsites')
            n = int(name[i + 9:i + 11])
            if n == 1:
                n = int(name[i + 9:i + 11])
            i = name.find('temp')
            temp = float(name[i+5:i+9])
            d = 0
            energy_range = get_energy_range(n)
            energy_hist = np.zeros((len(energy_range)))

            # INITIALIZING HISTOGRAM BINS
            temp_num = energy_range[-1]
            energy_bins = energy_range - 0.5
            energy_bins = np.append(energy_bins, temp_num + 0.5)
            boundary_energy_len = 16 + (n-2)*12 + 1
            boundary_energy_hist = np.zeros(2*boundary_energy_len + 1)
            boundary_energy_range = np.linspace(-boundary_energy_len, boundary_energy_len, 2*boundary_energy_len + 1)
            temp_num = boundary_energy_range[-1]
            boundary_energy_bins = boundary_energy_range - 0.5
            boundary_energy_bins = np.append(boundary_energy_bins, temp_num + 0.5)
            joint_dist = np.zeros((len(energy_range), len(boundary_energy_range)))

            # INITIALIZING FILENAMES
            filename_sim = str(n).zfill(2) + 'x' + str(n).zfill(2) + '_T=' + str(temp) + '_p_sim.txt'
            filename_be = str(n).zfill(2) + 'x' + str(n).zfill(2) + '_T=' + str(temp) + '_p_boundary_energy.txt'
            filename_joint = str(n).zfill(2) + 'x' + str(n).zfill(2) + '_T=' + str(temp) + '_joint_dist.txt'
            filename_mutual_inf = str(n) + 'x' + str(n) + '_mutual_inf.txt'

            if load_data == "true":
                new_path = path + '/' + name
                for root, dirs, files in os.walk(new_path):
                    for directory in dirs:
                        if directory != name:
                            for filename in os.listdir(new_path + '/' + directory):
                                if filename[-4:-1] == '.tx':
                                    sample_energy_arr, boundary_energy = iter_loadtxt(new_path + '/' + directory + '/' + filename, 'double', 1, 2)
                                    hist, edges = np.histogram(sample_energy_arr, energy_bins)
                                    energy_hist += hist
                                    b_hist, edges = np.histogram(boundary_energy, boundary_energy_bins)
                                    boundary_energy_hist += b_hist
                                    for i in range(len(sample_energy_arr)):
                                        ([idx_1],) = np.where(boundary_energy[i] == boundary_energy_range)
                                        ([idx_2],) = np.where(sample_energy_arr[i] == energy_range)
                                        joint_dist[idx_2][idx_1] += 1

                for filename in os.listdir(new_path):
                    if filename[-4:-1] == '.tx':

                        sample_energy_arr, boundary_energy = iter_loadtxt(new_path + '/' + filename, 'double', 1, 2)
                        hist, edges = np.histogram(sample_energy_arr, energy_bins)
                        energy_hist += hist
                        b_hist, edges = np.histogram(boundary_energy, boundary_energy_bins)
                        boundary_energy_hist += b_hist
                        for i in range(len(sample_energy_arr)):
                            ([idx_1],) = np.where(boundary_energy[i] == boundary_energy_range)
                            ([idx_2],) = np.where(sample_energy_arr[i] == energy_range)
                            joint_dist[idx_2][idx_1] += 1
                            d += 1

                os.chdir(system_energy_path + str(n).zfill(2))
                p_sim = energy_hist/sum(energy_hist)
                total_samples = sum(energy_hist)
                p_boundary_energy = boundary_energy_hist/sum(boundary_energy_hist)
                joint_dist *= 1/total_samples

                fit_len = 0
                for i in p_sim:
                    if i < 1e-6:
                        i = 0
                    if i != 0:
                        fit_len += 1

                c_sim = 0
                if c_sim == 0:
                    with open(filename_sim, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(energy_range)
                    c_sim += 1

                with open(filename_sim, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(p_sim)
                os.chdir(boundary_energy_path + str(n).zfill(2))
                c_be = 0
                if c_be == 0:
                    with open(filename_be, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(boundary_energy_range)
                    c_be += 1

                with open(filename_be, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(p_boundary_energy)

                os.chdir(joint_energy_path + str(n).zfill(2))
                c_joint = 0
                if c_joint == 0:
                    with open(filename_joint, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(energy_range)
                    c_joint += 1

                with open(filename_joint, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(boundary_energy_range)
                    csv_writer.writerow([total_samples])
                    csv_writer.writerows(joint_dist)

            else:
                os.chdir(system_energy_path + str(n).zfill(2))
                p_sim = []

                for i in range(len(energy_range)):
                    [p_i] = iter_loadtxt(filename_sim, "single", i, 0, delimiter='\t', skiprows=1, dtype=float)
                    p_sim = np.append(p_sim, p_i)

                fit_len = 0
                for i in p_sim:
                    if i < 1e-7:
                        i = 0
                    if i != 0:
                        fit_len += 1
                fit_len = fit_len

                i = -1
                j = -1
                counter_1 = -1
                os.chdir(joint_energy_path + str(n).zfill(2))
                with open(filename_joint, mode='r') as file:
                    csv_reader = csv.reader(file, delimiter='\t')
                    for row in csv_reader:
                        counter_1 += 1
                        if counter_1 == 2:
                            for k in row:
                                total_samples = float(k)
                        if counter_1 == 1:
                            const = 0
                            for k in row:
                                boundary_energy_range[const] = float(k)
                                const += 1
                        if counter_1 > 2:
                            i += 1
                            j = -1
                            for k in row:
                                j += 1
                                joint_dist[i][j] = float(k)

                os.chdir(boundary_energy_path + str(n).zfill(2))
                counter_2 = -1
                p_boundary_energy = []
                with open(filename_be, mode='r') as file:
                    csv_reader = csv.reader(file, delimiter='\t')
                    for row in csv_reader:
                        counter_2 += 1
                        if counter_2 == 1:
                            for k in row:
                                p_boundary_energy = np.append(p_boundary_energy, float(k))

            mutual_inf = 0
            for i in range(len(energy_range)):
                for j in range(len(boundary_energy_range)):
                    if joint_dist[i][j] != 0 and p_sim[i] != 0 and p_boundary_energy[j] != 0:
                        mutual_inf += joint_dist[i][j]*np.log(joint_dist[i][j]/(p_sim[i]*p_boundary_energy[j]))

            os.chdir(plots_and_figs_path + str(n).zfill(2))

            if c_mutual_inf == 0:
                with open(filename_mutual_inf, mode='w') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow([temp])
                    csv_writer.writerow([mutual_inf])

            if c_mutual_inf != 0:
                with open(filename_mutual_inf, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow([temp])
                    csv_writer.writerow([mutual_inf])
            c_mutual_inf += 1

            if n <= 4:

                # CANONICAL FITTING
                x = p_sim*total_samples
                x = x[0:fit_len]

                betas = np.linspace(0.0001, 2, 1000)
                energy_ss, p_ss, energy_counts_ss = small_site_dist(n)
                _, p_sim, energy_ss, p_ss = force_match(energy_ss, p_ss, energy_range, p_sim, n)
                p, beta_canonical = match_avg_energy(energy_range, p_sim, energy_range, p_ss, betas)

                can_max_log_l = log_factorial(total_samples)
                can_max_log_l -= sum(log_factorial(x))
                can_max_log_l += x @ np.log(p[0:fit_len].T)

            # FIGURE PARAMS
                os.chdir(plots_and_figs_path + str(n).zfill(2))
                fig, ax = plt.subplots(figsize=(5, 4), dpi=800)
                ax.plot(energy_range[0:fit_len], (p[0:fit_len]), marker='o', markersize=6, linewidth=0, label=r'Canonical')
                ax.plot(energy_range[0:fit_len], (p_sim[0:fit_len]), marker='o', markersize=6, linewidth=0, label=r'Simulated')
                ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
                ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')
                ax.set_ylim((10**-7.5, 1))
                ax.set_yscale('log')
                ax.set_xlabel(r'Energy ($\epsilon$)',  size=18)
                ax.set_ylabel(r'$p(\epsilon)$', size=18)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)
                for tick in ax.xaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                plt.xticks(rotation=45)
                plt.title(str(n) + "x" + str(n) + ', T = ' + str(temp) + ' Energy PDF', fontsize=18)
                plt.tight_layout()
                plt.legend(loc='lower left', fontsize=14)
                plt.savefig(str(n) + "x" + str(n)+"_site_" + "T=" + str(temp) + "_canonical_fit.jpg")
                print('saved plot')

                # COMPARE VARIENCE PER SPIN SITE
                var_p_sim = np.sqrt(np.var(p_sim[0:fit_len]))/n
                var_p_fit_canonical = np.sqrt(np.var(p[0:fit_len]))/n
                diff_canonical = abs(var_p_sim - var_p_fit_canonical)
                KL_div = KL_divergence(np.asarray(p_sim), np.asarray(p[0:fit_len]))

                # SAVING CANONICAL DATA TO CSV FILE
                data = [n, 1/beta_canonical, temp,  var_p_fit_canonical, var_p_sim, diff_canonical, KL_div]
                filename_1 = str(n) + "x" + str(n) + "_canonical_data.txt"
                header = ["site_size", "fitted_temp", "initial_temp", "var_fit", "var_sim", "var_diff", "KL_div"]
                if c_1 == 0:
                    with open(filename_1, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(header)
                    c_1 += 1

                with open(filename_1, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(data)

                # SUPERSTATISTICAL FITTING
                beta_arr = np.linspace(0.001, 5, 100)
                error = 100
                avg_e_sim = sum(p_sim[0:fit_len]*energy_range[0:fit_len])

                p_1 = load_bayesian(n, energy_range, bayesian_array_path)
                p_1 = np.transpose(p_1)
                super_max_log_l = -1e50
                for alpha in np.logspace(np.log10(n), np.log10(5*n**3), 1500):
                    lam_arr = np.logspace(np.log10(alpha/5), np.log10(5/4*alpha), 1500)
                    for lam in lam_arr:
                        arr, p_1, p_beta = small_sites_get_fit(beta_arr, p_1, lam, alpha)
                        p_fit_1 = np.transpose(arr/np.sum(arr))
                        if fit_type == "KL":
                            error_temp = KL_divergence(p_sim[0:fit_len], p_fit_1[0:fit_len])
                        if error_temp < error:
                            error = error_temp
                            var_p_fit_super = np.sqrt(np.var(p_fit_1))/n
                            diff_super = abs(var_p_sim - var_p_fit_super)
                            lam_0 = lam
                            alpha_0 = alpha

                        log_l = log_factorial(total_samples)
                        log_l -= sum(log_factorial(x))
                        log_l += x @ np.log(p_fit_1[0:fit_len].T)
                        if log_l > super_max_log_l:
                            super_max_log_l = log_l
                            p_max_l = p_fit_1

                arr, p_1, p_beta = small_sites_get_fit(beta_arr, p_1, lam_0, alpha_0)
                p_fit = np.transpose(arr/np.sum(arr))

                os.chdir(plots_and_figs_path + str(n).zfill(2))
                fig, ax = plt.subplots(figsize=(5, 4), dpi=800)
                ax.plot(energy_range[0:fit_len], (p_fit[0:fit_len]), marker='o', markersize=6, linewidth=0, label=r'Superstatistical')
                ax.plot(energy_range[0:fit_len], (p_sim[0:fit_len]), marker='o', markersize=6, linewidth=0, label=r'Simulated')
                ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
                ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')
                ax.set_ylim((10**-7.5, 1))
                ax.set_yscale('log')
                ax.set_xlabel(r'Energy ($\epsilon$)',  size=18)
                ax.set_ylabel(r'$p(\epsilon)$', size=18)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)
                for tick in ax.xaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                plt.xticks(rotation=45)
                plt.title(str(n) + "x" + str(n) + ', T = ' + str(temp) + ' Energy PDF', fontsize=18)
                plt.tight_layout()
                plt.legend(loc='lower left', fontsize=14)
                plt.savefig(str(n) + "x" + str(n) + "_site_" + "T=" + str(temp) + "_superstatistical_" + str(fit_type) + "_fit.jpg")
                print('saved plot')

                # DATA ANALYSIS QUANTITIES
                var_p_fit_super = np.sqrt(np.var(p_fit[0:fit_len]))/n
                diff_super = abs(var_p_sim - var_p_fit_super)
                KL_div = KL_divergence(np.asarray(p_sim), np.asarray(p_fit[0:fit_len]))
                mu_beta = lam_0/alpha_0
                std_beta = (lam_0**2/(alpha_0 - 1)**2/(alpha_0-2))**0.5
                coeff_var_beta = std_beta/mu_beta

                data = [n, temp,  var_p_fit_super, var_p_sim, diff_super, KL_div, coeff_var_beta, lam_0, alpha_0]
                filename_2 = str(n) + "x" + str(n) + "_superstatistcal_data_" + str(fit_type) + ".txt"
                header = ["site_size", "intialized_temp", "var_fit", "var_sim", "var_diff", "KL_div", "coeff_var", "lambda", "alpha"]

                if c_2 == 0:
                    with open(filename_2, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(header)
                    c_2 += 1

                with open(filename_2, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(data)

                data_likelihood = [n, temp, can_max_log_l, super_max_log_l, -2**0.5*(can_max_log_l - super_max_log_l)]
                filename_likelihood = str(n) + "x" + str(n) + "_likelhood_data_" + str(fit_type) + ".txt"
                header = ["site_size", "init_temp", "canonical max log likelihood", "super max log likelihood", "likelihood test statistic"]
                if c_l == 0:
                    with open(filename_likelihood, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(header)
                    c_l += 1

                with open(filename_likelihood, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(data_likelihood)

            if n > 4:
                x = p_sim*total_samples
                x = x[0:fit_len]

                var_p_sim = np.sqrt(np.var(p_sim[0:fit_len]))/n
                os.chdir(bayesian_array_path + str(n).zfill(2))
                filename_bayes = "bayesian_array_n=" + str(n).zfill(2) + "_final_1.txt"
                arr = iter_loadtxt(filename_bayes, 'single', 0, 0)
                L = len(arr)
                beta_arr = np.linspace(0.0001, 5, len(arr))
                p_1 = np.zeros((len(arr), len(energy_range)))
                p_beta = np.zeros((len(arr), 1))

                for i in range(len(energy_range)):
                    filename = "bayesian_array_n=" + str(n).zfill(2) + "_final_1.txt"
                    arr = iter_loadtxt(filename, 'single', i, 0)
                    for j in range(L):
                        p_1[j][i] = arr[j]
                (r, c) = np.shape(p_1)
                for row in range(r):
                    Z = 0
                    for col in range(c):
                        Z += p_1[row][col]
                for col in range(c):
                    p_1[row][col] = p_1[row][col]/Z
                p_1 = np.transpose(p_1)


                error = 100
                p_1_canonical = p_1.T
                (r, c) = np.shape(p_1_canonical)

                for row in range(r):
                    p_canonical = p_1_canonical[row][:]
                    # print(p_canonical)
                    beta = (5 - 0.01)/500*row
                    error_temp = (sum(p_canonical[0:fit_len]*energy_range[0:fit_len]) - sum(p_sim[0:fit_len]*energy_range[0:fit_len]))**2
                    if error_temp < error:
                        error = error_temp
                        best_p_can = p_canonical
                        var_p_fit_can = np.sqrt(np.var(p_canonical[0:fit_len]))/n
                        diff_can = abs(var_p_sim - var_p_fit_can)
                        print("error_temp = ", error_temp)
                        print("diff_super = " + str(diff_can))
                        min_beta = beta
                        print(min_beta)
                l_can = 0
                l_sim = 0
                for i in best_p_can:
                    if i != 0:
                        l_can += 1
                for i in p_sim:
                    if i != 0:
                        l_sim += 1
                if l_can <= l_sim:
                    l = l_can
                else:
                    l = l_sim

                os.chdir(plots_and_figs_path + str(n).zfill(2))
                fig, ax = plt.subplots(figsize=(5,4), dpi=800)
                ax.plot(energy_range[0:l], (best_p_can[0:l]), marker='o', markersize=4, linewidth=0, label='Canonical')
                ax.plot(energy_range[0:l], (p_sim[0:l]), marker='o', markersize=4, linewidth=0, label='Simulated')
                ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
                ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')
                ax.set_ylim((10**-7.5, 1))
                ax.set_yscale('log')
                ax.set_xlabel(r'Energy ($\epsilon$)',  size=18)
                ax.set_ylabel(r'$p(\epsilon)$', size=18)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)
                for tick in ax.xaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                plt.xticks(rotation=45)
                plt.title(str(n) + "x" + str(n) + ', T = ' + str(temp) + ' Energy PDF', fontsize=18)
                plt.tight_layout()
                plt.legend(loc='lower left', fontsize=14)
                plt.savefig(str(n) + "x" + str(n)+"_site_" + "T=" + str(temp) + "_canonical_" + str(fit_type) + "_fit.jpg", dpi=800)
                print('saved plot')

                can_max_log_l = log_factorial(total_samples)
                for i in range(len(x)):
                    if x[i] != 0 and best_p_can[i]:
                        can_max_log_l -= log_factorial(x[i])
                        can_max_log_l += x[i]*np.log(best_p_can[i])

                var_p_fit_can = np.sqrt(np.var(best_p_can[0:fit_len]))/n
                diff_can = abs(var_p_sim - var_p_fit_can)
                KL_div = KL_divergence(np.asarray(p_sim[0:fit_len]), np.asarray(best_p_can[0:fit_len]))

                #SAVING DATA TO CSV FILE
                data = [n, 1/min_beta, temp,  var_p_fit_can, var_p_sim, diff_can, KL_div]
                header = ["site_size", "fitted_temp", "initial_temp", "var_fit", "var_sim", "var_difference", "KL_div"]
                if large_sites_canonical == "true":
                    filename_1 = str(n) + "x" + str(n) + "_canonical_data_" + fit_type + ".txt"

                if c_1 == 0:
                    with open(filename_1, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(header)
                    c_1 += 1

                with open(filename_1, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(data)

                super_max_log_l = -1e500

                error = 100
                for alpha in np.logspace(np.log10(n), np.log10(5*n**3), 1500):
                    for lam in np.logspace(np.log10(2*alpha/7), np.log10(alpha + alpha/4), 1500):
                        p_beta = invgamma.pdf(beta_arr, alpha, loc=0, scale=lam)
                        arr = np.matmul(p_1, p_beta)
                        p_fit_1 = np.transpose(arr/np.sum(arr))

                        if fit_type == "var" or fit_type == "var_limited":
                            error_temp = (np.var(p_sim[1:fit_len])**0.5 - np.var(p_fit_1[1:fit_len])**0.5)**2 + \
                                (sum(p_fit_1[0:fit_len]*energy_range[0:fit_len]) - sum(p_sim[0:fit_len]*energy_range[0:fit_len]))**2
                        if fit_type == "KL":
                            error_temp = KL_divergence(np.asarray(p_sim[0:fit_len]), np.asarray(p_fit_1[0:fit_len]))

                        if error_temp < error:
                            error = error_temp
                            print("error = " + str(error))
                            var_p_fit_super = np.sqrt(np.var(p_fit_1[0:fit_len]))/n
                            diff_super = abs(var_p_sim - var_p_fit_super)
                            print("diff_super = " + str(diff_super))
                            lam_0 = lam
                            alpha_0 = alpha

                        log_l = log_factorial(total_samples)
                        log_l -= sum(log_factorial(x))
                        log_l += x @ np.log(p_fit_1[0:fit_len].T)
                        if log_l > super_max_log_l:
                            super_max_log_l = log_l
                            p_max_l = p_fit_1

                arr, p_1, p_beta = large_sites_get_fit(energy_range, n, lam_0, alpha_0, 100, L, beta_arr, bayesian_array_path)
                p_fit_super = np.transpose(arr/np.sum(arr))

                os.chdir(plots_and_figs_path + str(n).zfill(2))
                fig, ax = plt.subplots(figsize=(5, 4), dpi=800)
                ax.plot(energy_range[0:fit_len], (p_fit_super[0:fit_len]), marker='o', markersize=4, linewidth=0, label='Superstatistical')
                ax.plot(energy_range[0:fit_len], (p_sim[0:fit_len]), marker='o', markersize=4, linewidth=0, label='Simulated')
                ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
                ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')
                ax.set_ylim((10**-7.5, 1))
                ax.set_yscale('log')
                ax.set_xlabel(r'Energy ($\epsilon$)',  size=18)
                ax.set_ylabel(r'$p(\epsilon)$', size=18)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)
                for tick in ax.xaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_fontsize(18)
                    tick.set_fontname('serif')
                plt.xticks(rotation=45)
                plt.title(str(n) + "x" + str(n) + ', T = ' + str(temp) + ' Energy PDF', fontsize=18)
                plt.tight_layout()
                plt.legend(loc='lower left', fontsize=14)
                plt.savefig(str(n) + "x" + str(n)+"_site_" + "T=" + str(temp) + "_superstatistical_" + str(fit_type) + "_fit.jpg", dpi=800)
                print('saved plot')

                var_p_sim = np.sqrt(np.var(p_sim[0:fit_len]))/n
                var_p_fit_super = np.sqrt(np.var(p_fit_super[0:fit_len]))/n
                diff_super = abs(var_p_sim - var_p_fit_super)
                mu_beta = lam_0/(alpha_0)
                std_beta = (lam_0**2/(alpha_0 - 1)**2/(alpha_0-2))**0.5
                coeff_var_beta = std_beta/mu_beta
                KL_div = KL_divergence(np.asarray(p_sim[0:fit_len]), np.asarray(p_fit_super[0:fit_len]))
                norm_KL = KL_div/np.log10(fit_len)

                data = [n, temp,  var_p_fit_super, var_p_sim, diff_super, KL_div, coeff_var_beta, lam_0, alpha_0]
                filename_2 = str(n) + "x" + str(n) + "_superstatistcal_data_" + str(fit_type) + ".txt"
                header = ["site_size", "init_temp", "var_fit", "var_sim", "var_diff", "KL_div", "coeff_var", "lambda", "alpha"]

                if c_2 == 0:
                    with open(filename_2, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(header)
                    c_2 += 1

                with open(filename_2, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(data)

                data_likelihood = [n, temp, can_max_log_l, super_max_log_l, -2**0.5*(can_max_log_l - super_max_log_l)]
                filename_likelihood = str(n) + "x" + str(n) + "_likelhood_data_" + str(fit_type) + ".txt"
                header = ["site_size", "init_temp", "canonical max log likelihood", "super max log likelihood", "likelihood test statistic"]
                if c_l == 0:
                    with open(filename_likelihood, mode='w') as file:
                        csv_writer = csv.writer(file, delimiter='\t')
                        csv_writer.writerow(header)
                    c_l += 1

                with open(filename_likelihood, mode='a') as file:
                    csv_writer = csv.writer(file, delimiter='\t')
                    csv_writer.writerow(data_likelihood)

            if temp == 2.8:
                d = 1
                break
