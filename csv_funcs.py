# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:06:01 2020

@author: Lukas Herron
"""
import csv
import os
import numpy as np

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return 'true'
        else:
            return 'false'

def getnum(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            i = name.find('num')
            testnum = int(name[i+4:i+8])
            if testnum > num:
                num = testnum
    return num

def mkdir(initial_dtype, initial_value,  N , n, subsites):
    data_dir = '/ufrc/pdixit/lukasherron/ising_data'
    current_dir = '/ufrc/pdixit/lukasherron/ising_sim'
    os.chdir(data_dir)
    num = getnum(data_dir) + 1
    new_dir = str(initial_dtype) + '=' + str(initial_value) + \
        '_lattice='+str(N) + 'x' + str(N) + '_subsites=' + str(subsites) \
        + '_samples=' + str(n) + '_num=' +  str(num).zfill(4)
    os.mkdir(new_dir)
    os.chdir(current_dir)
    return new_dir

def create_file(init_type, init_value , N, n, subsites, num):
    c = 0
    while c == 0:
        data_dir = '/ufrc/pdixit/lukasherron/ising_data'
        current_dir = '/ufrc/pdixit/lukasherron/ising_sim'
        os.chdir(data_dir)
        
        filename = str(init_type) + '=' + str(init_value) + \
        '_lattice='+str(N) + 'x' + str(N) + '_subsites=' + str(subsites) \
        + '_samples=' + str(n) + '_num=' +  str(num).zfill(4)+'.txt'
        if find(filename, data_dir) == 'false':
            c += 1
            os.chdir(current_dir)
            return filename
        

def write_to_file(filename, data, new_dir):
    data_dir = '/ufrc/pdixit/lukasherron/ising_data' +str(new_dir)
    current_dir = '/ufrc/pdixit/lukasherron/ising_sim'
    os.chdir(data_dir)
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter = '\t')
        writer.writerows(data)
    os.chdir(current_dir)        

def main(data, init_type, init_value, N, n, subsites):
    filename = create_file(init_type, init_value , N, n, subsites)
    write_to_file(filename, data)
    
def loaddata(path, col):
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
    n = len(data1)
    master_arr = np.zeros((n,5))
    for i in range(len(data1)):
        master_arr[i][0] = data1[i]
        master_arr[i][1] = data2[i]
        master_arr[i][2] = data3[i]
        master_arr[i][3] = data4[i]
        master_arr[i][4] = data5[i]
    return master_arr


    
    
        
        
    

