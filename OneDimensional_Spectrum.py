import numpy as np
from matplotlib import pyplot as plt
from elfouhaily import *

''' Compares the 1D and 2D formulations for both the case with and without 
capillary waves.

Spectrum inputs can be copy and pasted from "Synthesize_Images.py"
'''
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
N     = int(8192)        # number of point for FFTs
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

################################################################################
## 1D CALCS
# constants and empirical parameters
g=9.81                   # acceleration of gravity: meters/second/second
# k0=g/W**2              # reference wave number;
kp=.86*g/W**2            # wavenumber at spectral peak
cp=np.sqrt(g/kp)         # phase speed at spectral peak
cm=.2305                 # phase speed at gravity/capillary crossover
km=g/cm**2               # wavenumber at gravity/capillary crossover
om=W/cp                  # inverse wave age;
omc=om*np.cos(agetheta)  # directional inverse wave age
alfp=6e-3*np.sqrt(om)    # Phillips-Kitaigorodskii parameter (dimensionless)
sig=0.08*(1+4/omc**3) 
ustar=.025*W             # ustar=.1*W

if omc<1:
    gam=1.7
else:
    gam=1.7+6*np.log10(omc);

if ustar<cm:
    alfm=1e-2*(1+np.log(ustar/cm))
else:
    alfm=1e-2*(1+3*np.log(ustar/cm))

LPM=np.exp(-(5/4)*(kp/k)**2)
GAM=np.exp(-(np.sqrt(k/kp)-1)**2/(2*sig**2))
Jp=gam**GAM
F3=np.exp(-(om/np.sqrt(10))*(np.sqrt(k/kp)-1))
Blow=.5*alfp*np.sqrt(k/kp)*LPM*Jp*F3
Bhigh=.5*alfm*np.sqrt(k/km)*np.exp(-(1/4)*((k/km)-1)**2)
Bhigh=Bhigh*LPM

################################################################################
## 2D CALCS
# Efouhaily spectrum with capillary waves
(KX,KY,KSPEC,offset,varout) = elfouhaily_varspec(W, windaz, agetheta, 
    dx, N, alftemp, cap=True, cap_factor=cap_factor)
dk = KX[0,1]-KX[0,0]
S_dft = cont1sVar_to_disc2sAmp(dk, KSPEC, debug=debug, offset=offset, 
    hermitian_option=hermitian_option, seed_flag=True, seed=0)
SX, SY = elev_to_slope(KX, KY, S_dft, offset=offset)

# sea state realization with capillary waves
X,Y = freq_to_coords(N, dk)
SSE = spec_to_sse(S_dft, debug=debug)
slope_y = spec_to_sse(SY, debug=debug)
slope_x = spec_to_sse(SX, debug=debug)  
if slope_option:
    slope_x = rad_to_slope(slope_x)
    slope_y = rad_to_slope(slope_y)

# Efouhaily spectrum with no capillary waves (NC stands for no capillaries)
(KX,KY,KSPEC_NC,offset,varout_NC) =  elfouhaily_varspec(W, windaz, agetheta, 
    dx, N, alftemp, cap=False, cap_factor=cap_factor)
S_dft_NC = cont1sVar_to_disc2sAmp(dk, KSPEC_NC, debug=debug, offset=offset, 
    hermitian_option=hermitian_option, seed_flag=True, seed=0)
SX_NC, SY_NC = elev_to_slope(KX, KY, S_dft_NC, offset=offset)

# sea state realization with no capillary waves
SSE_NC = spec_to_sse(S_dft_NC, debug=debug)
slope_y_NC = spec_to_sse(SY_NC, debug=debug)
slope_x_NC = spec_to_sse(SX_NC, debug=debug)  
if slope_option:
    slope_x_NC = rad_to_slope(slope_x_NC)
    slope_y_NC = rad_to_slope(slope_y_NC)

################################################################################
## PLOT
ii = int(N/2)                                         # middle of 2d Elfouhaily Spectrum
KSPEC = KSPEC*np.sqrt(KX**2+KY**2)/varout[3]          # simplify 2D form into 1D
KSPEC_NC = KSPEC_NC*np.sqrt(KX**2+KY**2)/varout_NC[3] # simplify 2D form into 1D

plt.figure()
ax = plt.gca()
plt.loglog(k,(Blow + Bhigh)/k**3,'b-')
plt.loglog(k,(Blow + cap_factor*Bhigh)/k**3,'r-')
plt.loglog(KX[ii,:], KSPEC[ii,:],'bo')
plt.loglog(KX[ii,:], KSPEC_NC[ii,:],'ro')
plt.grid(which="major", axis="both", ls='-', color='k', alpha=0.25)
plt.grid(which="minor", axis="both", ls=':', color='k', alpha=0.25)
xmin,xmax = (-3,4)
ymin,ymax = (-12,3)
xmajor = [1*10.**i for i in np.arange(xmin-1,xmax+2)]
ymajor = [1*10.**i for i in np.arange(ymin-1,ymax+2)]
xminor = []
for i in np.arange(xmin-1,xmax+2):
    xminor += [j*10.**i for j in np.arange(2,10)]
yminor = []
for i in np.arange(ymin-1,ymax+2):
    yminor += [j*10.**i for j in np.arange(2,10)]
ax.set_xticks(xmajor)
ax.set_xticks(xminor, minor = True)
ax.set_yticks(ymajor)
ax.set_yticks(yminor, minor = True)
ax.set_xlim([10**xmin, 10**xmax])
ax.set_ylim([10**ymin, 10**ymax])
plt.title('Elfouhaily Elevation Spectrum for U = {} Meters/Second'.format(W))
plt.xlabel('Wavenumber k: 1/meter')
plt.ylabel('Power Spectral Density')
plt.legend(['1D Original Spectrum','1D Reduced Capillary Action','2D Original Spectrum','2D Reduced Capillary Action'])

plt.show()