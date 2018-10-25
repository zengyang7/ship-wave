# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:24:04 2017

@author: adamwise
"""
import numpy as np
import os
#import matplotlib.pyplot as plt 
import time

from elfouhaily import *
from create_lines import *
from scipy import interpolate
from numpy import random

#############################
## IMAGE SYNTHEISIS INPUTS ## 
#############################

Nss = 1                # number of sea states
Nim_ss = 1              # number of images per sea state
Nl_max = 2              # max number of lines on an image
Np = 10000              # number of pixels in the x and y directions for the image (only squares)

#####################
## SPECTRUM INPUTS ## 
#####################

# environ params
W        = 10            # wind speed (meters/second)
windaz   = 135           # wind heading direction (degrees, from 0-180)
agetheta = 0             # angle between wind and dominant wave (degrees)
alftemp  = 0             # anglular param (radians)

# sensor & processing params (in this version, sensor and FFT spatial 
# res must be equal)
N     = int(100)        # number of point for FFTs
Lx    = 100.             # physical domain (meters)
dx    = Lx/N             # resolution in x-space (meters)

# satellite params
sat_interp_option = 1    # change resolution for the satellite image
sat_res = Lx/Np          # resolution in x-space for satellite images (meters)
wake_flag = 1            # 0 for no wake, 1 for wake
min_wake_m = 10          # minimum wake size (meters)
max_wake_m = 40          # maximum wake size (meters)

# geometry params (assumes range >> image footprint)
thetar   = 85            # sensor elevation angle (degrees)
phir     = -45           # sensor azimuth angle (degrees)
thetasun = 45            # solar elevation angle (degrees)
phisun   = 110           # solar azimuth angle (degrees)

# other options
hermitian_option = 2     # Hermitian, random phases.
slope_option = 1         # 0 for radians, 1 for dy/dx
debug = 0                # 0 for no debug, 1 for debug w/no plots, 2 for plots.
more_points = False      # for verification
f = 5                    # discretization factor for verification

################
## FILE PATHS ## 
################

# create a folder called images to save training images
file_path_im = './images_ss2'
if not os.path.exists(file_path_im):
    os.makedirs(file_path_im)
    
# create a folder called images to save the background image
file_path_ss = './seastate_ss2'
if not os.path.exists(file_path_ss):
    os.makedirs(file_path_ss)
    
# create a folder called images to save the corresponging theta and t coords, wake thickness, and sea state number
file_path_data = './data_ss1'
if not os.path.exists(file_path_data):
    os.makedirs(file_path_data)
    
 # create a folder called images to save the images with no capillary waves
file_path_NC = './images_NC'
if not os.path.exists(file_path_NC):
    os.makedirs(file_path_NC)   



tic = time.time()
x = np.arange(0,dx*N,dx)
for Nss in range(Nss):
    ################################################################################
    ## RUN
    # Efouhaily spectrum with capillary waves
    (KX,KY,KSPEC,offset,varout) = elfouhaily_varspec(W,windaz,agetheta,dx,N,alftemp,cap=True)
    dk = KX[0,1]-KX[0,0]
    S_dft = cont1sVar_to_disc2sAmp(dk, KSPEC, debug=debug, offset=offset, 
        hermitian_option=hermitian_option, seed_flag=True, seed=Nss)
    SX, SY = elev_to_slope(KX, KY, S_dft, offset=offset)

    
    
    # sea state realization with capillary waves
    X,Y = freq_to_coords(N, dk)
    #SSE = spec_to_sse(S_dft, debug=debug)
    slope_y = spec_to_sse(SY, debug=debug)
    slope_x = spec_to_sse(SX, debug=debug)  
    if slope_option:
        slope_x = rad_to_slope(slope_x)
        slope_y = rad_to_slope(slope_y)
        
    # Efouhaily spectrum with no capillary waves (NC stands for no capillaries)
    (KX,KY,KSPEC_NC,offset,varout) = elfouhaily_varspec(W,windaz,agetheta,dx,N,alftemp,cap=False)
    S_dft_NC = cont1sVar_to_disc2sAmp(dk, KSPEC_NC, debug=debug, offset=offset, 
        hermitian_option=hermitian_option, seed_flag=True, seed=Nss)
    
    
    SX_NC, SY_NC = elev_to_slope(KX, KY, S_dft_NC, offset=offset)
#    Remove short wave length waves
    Klim = 2
    K = np.sqrt(KX**2 + KY**2)
    SX_NC[K>Klim] *= 0;
    SY_NC[K>Klim] *= 0;
    
    # sea state realization with no capillary waves
    #SSE_NC = spec_to_sse(S_dft_NC, debug=debug)
    slope_y_NC = spec_to_sse(SY_NC, debug=debug)
    slope_x_NC = spec_to_sse(SX_NC, debug=debug)  
    if slope_option:
        slope_x_NC = rad_to_slope(slope_x_NC)
        slope_y_NC = rad_to_slope(slope_y_NC)
        
    ################################################################################
    ## INTERPOLATE FOR SEA STATE IMAGES
    # should we interpolate the sea state or the slope?
    # probably the slope
    
    if sat_interp_option == 1:
        f_slope_x = interpolate.RectBivariateSpline(X[0,:], Y[:,0], slope_x)
        f_slope_y = interpolate.RectBivariateSpline(X[0,:], Y[:,0], slope_y)
        f_slope_x_NC = interpolate.RectBivariateSpline(X[0,:], Y[:,0], slope_x_NC)
        f_slope_y_NC = interpolate.RectBivariateSpline(X[0,:], Y[:,0], slope_y_NC)
        
        Xsat = np.arange(0, Lx, sat_res)
        Ysat = np.arange(0, Lx, sat_res)
        
        slope_x_int = f_slope_x(Xsat, Ysat)
        slope_y_int = f_slope_y(Xsat, Ysat)
        slope_x_NC_int = f_slope_x_NC(Xsat, Ysat)
        slope_y_NC_int = f_slope_y_NC(Xsat, Ysat)

    (I, IMFFT) = sse_to_imspec(slope_x_int,slope_y_int,thetar,phir,thetasun,phisun)
    (I_NC, IMFFT_NC) = sse_to_imspec(slope_x_NC_int,slope_y_NC_int,thetar,phir,thetasun,phisun)

    # background image
    img = np.real(I)
    img[img > 1.] = 1.
    img = (img-np.min(img))/(np.max(img)-np.min(img))                       # normalize image
    img_NC = np.real(I_NC)
    img_NC[img_NC > 1.] = 1.
    if np.max(img_NC) == np.min(img_NC):
        img_NC *= 0
    else:
        img_NC = (img_NC-np.min(img_NC))/(np.max(img_NC)-np.min(img_NC))        # normalize image

    for Nim in range(Nim_ss): 
        image = np.copy(img)
        image_NC = np.copy(img_NC)
        if wake_flag == 1:
            wake_thickness_m = random.uniform(min_wake_m, max_wake_m)       # wake thickness in meters
            wake_thickness_p = int(wake_thickness_m/Lx*Np)                  # wake thickness in pixels
            image_w_lines, coords = add_lines(image, image_NC, wake_thickness_p, Nl_max)
        output = np.zeros([1,4])
        output[0,0] = coords[0,0]        # theta
        output[0,1] = coords[0,1]        # t
        output[0,2] = wake_thickness_m   # thickness in meters, could be changed to pixels with wake_thickness_p
        output[0,3] = Nss+1              # sea state number

        ## plot 
        #plt.figure(Nss*10 + Nim+1)
        #im = plt.imshow(image_w_lines, extent=(x[0],x[-1],x[0],x[-1]),cmap='gray')
        #plt.colorbar()
        
        # save
        if wake_flag == 1:
            np.savetxt(file_path_im + os.sep + 'train_ss_m' + str(Nss+1) + '_' + str(Nim+1), image_w_lines)
            np.savetxt(file_path_NC + os.sep + 'train_NC_m' + str(Nss+1) + '_' + str(Nim+1), image_NC)
            np.savetxt(file_path_data + os.sep + 'data_ss_m' + str(Nss+1) + '_' + str(Nim+1), output) # (theta, t, thickness (in meters, sea state number)
            if Nim == 0:
                np.savetxt(file_path_ss + os.sep + 'seastate_m' + str(Nss+1), image)             # background image
        else:
            np.savetxt(file_path_im + os.sep + 'train_ss_m' + str(Nss+1) + '_' + str(Nim+1), image)

               
    # keep track
    print("Sea State Realization Number: " + str(Nss+1))
    #print('Elapsed Time: {} s'.format(time.time()-tic))
