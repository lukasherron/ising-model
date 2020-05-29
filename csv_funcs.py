# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:06:01 2020

@author: Lukas Herron
"""

import csv
import os
import numpy as np



def find(name, path):
    '''
    Parameters
    ----------
    name : file to be seached for (string)
    path : path to search for file along (string)

    Returns
    -------
    True if file is located on path. False if not.

    '''
    for root, dirs, files in os.walk(path):
        if name in files:
            return 'true'
        else:
            return 'false'

def getnum(path):
    '''
    Parameters
    ----------
    path : path to check filenames along

    Returns
    -------
    num: An integer that allows identical simulations to be distinguished

    '''
    num = 0
    for root, dirs, files in os.walk(path):
        for name in dirs:
    
            i = name.find('num')
            testnum = int(name[i+4:i+8])
            if testnum > num:
                num = testnum
    return num

def mkdir(initial_dtype, initial_value,  N , n, subsites, data_dir, current_dir):
    '''
    

    Parameters
    ----------
    initial_dtype : Str "temp" or "mag" corresponding to how the lattice was initialized
    initial_value : Value of "temp" or "mag" the lattice was initalized to
    N : Size of NxN lattice
    n : Number of samples collected from lattice
    subsites : subsites = m is size of mxm small sampled sites

    Returns
    -------
    new_dir : creates a new directory with name according to convention:
            initial_dtype=initial_value_lattice=NxN_subsites=mxm_samples=n_num=xxxx
            using lattice and simulation properties.

    '''

    os.chdir(data_dir)
    num = getnum(data_dir) + 1
    new_dir = str(initial_dtype) + '=' + str(initial_value).zfill(4) + \
        '_lattice='+str(N) + 'x' + str(N) + '_subsites=' + str(subsites).zfill(4) \
        + '_samples=' + str(n) + '_num=' +  str(num).zfill(4)
    os.mkdir(new_dir)
    os.chdir(current_dir)
    return new_dir

def create_file(init_type, init_value , N, n, subsites, path):
    '''
    Parameters
    ----------
    init_type : Str "temp" or "mag" corresponding to how the lattice was initialized
    init_value : Value of "temp" or "mag" the lattice was initalized to
    n : Number of samples collected from lattice
    subsites : subsites = m is size of mxm small sampled sites
    num : Index for distinguishing identical simulations

    Returns
    -------
    filename : name of file to be saved according to
            initial_dtype=initial_value_lattice=NxN_subsites=mxm_samples=n_num=xxxx

    '''
    
    c = 0
    num = 0
    while c == 0:
        filename = str(init_type) + '=' + str(init_value).zfill(4) + \
        '_lattice='+str(N) + 'x' + str(N) + '_subsites=' + str(subsites).zfill(4) \
        + '_samples=' + str(n) + '_num=' +  str(num).zfill(4)+'.txt'
        if find(filename, path) == 'false':
            c += 1
        num += 1
        
    return filename
        

def write_to_file(filename, data):
    '''
    

    Parameters
    ----------
    filename : name of file to be written to
    data : 2D array returned by format_data(...)
    new_dir : suffix returned by mkdir(..) that creates a new directory according
                to simualtion properties

    Returns
    -------
    None. Writes data to .txt file

    '''
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter = '\t')
        writer.writerows(data)

def loaddata(path, col):
    '''
    

    Parameters
    ----------
    path : String that provides path to .txt file to be read from
    col : Column to read from 9strating from 0) in .txt file. Columns are according
            to formatting in format_data(...)

    Returns
    -------
    data : 1D array of data correspong to one of the arguments of format_data(...)

    '''
    l = 1000
    data, hist, energy_dist = [], np.zeros(l-1), []
    cwd = os.getcwd()
    os.chdir(path)
    counter = 0
    bins = []
    for i in range(l):
        bins = np.append(bins, -l/2 +0.5 + i)
        
    for root, dirs, filename in os.walk(path):
        files = filename
        for i in files:
            with open(str(i), mode = 'r') as file:
                single_data = csv.reader(file, delimiter='\t')
                for k in single_data:
                    if k[col] != "---" :
                        data = np.append(data, float(k[col]))
                        counter += 1
                        if counter == 10000:
                            for i in data:
                                c = 0
                                for j in energy_dist:
                                    if i == j:
                                        c += 1
                                if c == 0:
                                    energy_dist = np.append(energy_dist, i)
                            temp_hist, edges = np.histogram(data, bins)
                            hist += temp_hist
                            counter = 0
                            data = []
    energy_dist = np.sort(energy_dist)
    os.chdir(cwd)
    return hist, energy_dist

def format_data(data1, data2, data3, data4, data5, data6):
    '''
    

    Parameters
    ----------
    data1 : 1D array of data
    data2 : 1D array of data
    data3 : 1D array of data
    data4 : 1D array of data
    data5 : 1D array of data

    Returns
    -------
    master_arr : for m input data sets of length n, all are read into a mxn 2D array

    '''
    len_arr = []
    len_arr = np.append(len_arr, len(data1))
    len_arr = np.append(len_arr, len(data2))
    len_arr = np.append(len_arr, len(data3))
    len_arr = np.append(len_arr, len(data4))
    len_arr = np.append(len_arr, len(data5))
    len_arr = np.append(len_arr, len(data6))
    n = int(max(len_arr))
    master_arr = ["---"]*6*n
    master_arr = np.reshape(master_arr, (n,6))

    for i in range(len(data1)):
        master_arr[i][0] = str(data1[i])
    for i in range(len(data2)):
        master_arr[i][1] = str(data2[i])
    for i in range(len(data3)):
        master_arr[i][2] = str(data3[i])
    for i in range(len(data4)):
        master_arr[i][3] = str(data4[i])
    for i in range(len(data5)):
        master_arr[i][4] = str(data5[i])
    for i in range(len(data6)):

        master_arr[i][4] = str(data6[i])
        
    return master_arr
        



    
    
        
        
    
