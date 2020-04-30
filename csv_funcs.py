# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:06:01 2020

@author: Lukas Herron
"""
import csv
import os
import numpy as np



 data_dir = '/ufrc/pdixit/lukasherron/ising_data' +str(new_dir)
 current_dir = '/ufrc/pdixit/lukasherron/ising_sim'
    
    
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
        for name in files:
            i = name.find('num')
            testnum = int(name[i+4:i+8])
            if testnum > num:
                num = testnum
    return num

def mkdir(initial_dtype, initial_value,  N , n, subsites):
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
    new_dir = str(initial_dtype) + '=' + str(initial_value) + \
        '_lattice='+str(N) + 'x' + str(N) + '_subsites=' + str(subsites) \
        + '_samples=' + str(n) + '_num=' +  str(num).zfill(4)
    os.mkdir(new_dir)
    os.chdir(current_dir)
    return new_dir

def create_file(init_type, init_value , N, n, subsites, num):
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
    while c == 0:
        os.chdir(data_dir)
        
        filename = str(init_type) + '=' + str(init_value) + \
        '_lattice='+str(N) + 'x' + str(N) + '_subsites=' + str(subsites) \
        + '_samples=' + str(n) + '_num=' +  str(num).zfill(4)+'.txt'
        if find(filename, data_dir) == 'false':
            c += 1
            os.chdir(current_dir)
            return filename
        

def write_to_file(filename, data, new_dir):
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
    data_dir_1 = data_dir +str(new_dir)
    os.chdir(data_dir_1)
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter = '\t')
        writer.writerows(data)
    os.chdir(current_dir)        

    
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
    data = []
    cwd = os.getcwd()
    os.chdir(path)
    for filename in os.walk(path):
        with open(filename, mode = 'r') as file:
            single_data = np.loadtxt(file, delimiter='\t', usecols = col, unpack=True)
            data = np.append(data, single_data)
    os.chdir(cwd)
    return data

def format_data(data1, data2, data3, data4, data5):
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
    n = len(data1)
    master_arr = np.zeros((n,5))
    for i in range(len(data1)):
        master_arr[i][0] = data1[i]
        master_arr[i][1] = data2[i]
        master_arr[i][2] = data3[i]
        master_arr[i][3] = data4[i]
        master_arr[i][4] = data5[i]
    return master_arr


    
    
        
        
    

