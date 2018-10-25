#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:20:59 2018

@author: zengyang
"""

import numpy as np
import matplotlib.pyplot as plt
import os, time, random
## classes for discriminating
classes = {'seastate_ss2','images_ss2','images_NC'}

N     = int(100)        # number of point for FFTs
Lx    = 100.             # physical domain (meters)
dx    = Lx/N             # resolution in x-space (meters)

x = np.arange(0,dx*N,dx)


#### read data from the path
cwd = '/Users/zengyang/research/Final_Wake_Files/'

figuresavepath = '/Users/zengyang/research/Final_Wake_Files/fig/'
if not os.path.exists(figuresavepath):
    os.makedirs(figuresavepath)


for index, name in enumerate(classes):
    figpathsaves = figuresavepath+name+'/'
    if not os.path.exists(figpathsaves):
        os.makedirs(figpathsaves)
    t = 0
    class_path = cwd+name+'/'
    for data_name in os.listdir(class_path):
        t += 1
#        if t>9:
#            break
        if data_name == '.DS_Store':
            continue
        data_path = class_path + data_name
        data = np.loadtxt(data_path)
        plt.figure(1)
        im = plt.imshow(data, extent=(x[0],x[-1],x[0],x[-1]),cmap='gray')
        plt.colorbar()
        namesavefig = figpathsaves+data_name+'.pdf'
        plt.savefig(namesavefig)
        plt.close()
        