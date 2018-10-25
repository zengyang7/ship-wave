
''' Functions for creating synthetic SAR images of random sea states using 
the Elfouhaily spectrum. 

Functions provided by Dr. Eric Paterson. 
Originally created by ECH (2010) and revised by John G Pierce (2010). 


Translated to Python (from MATLAB) by Carlos Michelen-Strofer (2017).
Other modifications by Carlos Michelen-Strofer:
    * The function 'sse-kspec' now outputs the offsets and the the components 
        of KSPEC.
    * The Nyquist frequencies are now negative (NumPy format).

TODO: 
    * Handle singularity differently (no offsets).
    * Move matrix rotation function outside of sse_kspec
'''

import numpy as np
import scipy as sp

def sse_kspec(W, windaz, agetheta, dx, NFFT, alftemp, cap=True, cap_factor = 0.0, nyquist='negative'):
    '''Function last revised 9-AUG-2010, ECH
    Model last revised 1 June 2010, John G Pierce
    this code calculates the Elfouhaily spectrum of the sea surface elevation
    for a given wind speed and direction.
    Function Outputs: KX,KY are aligned with world local Cartesian coords (East 
    and North), the same system as a properly orthorectified image.
    KXR,KYR are rotated to align with wind heading direction.
    KSPEC and KSPECA are Elfouhaily power- and amplitude- spectral densities
    respectively. 
    '''
    # constants and empirical parameters
    d2rad = np.pi/180
    # shift wind azimuth by 90 degrees to match the "heading" convention,
    # clockwise azimuth angle from North:
    windaz = d2rad*(windaz+90) 
    agetheta = d2rad*agetheta
    g = 9.81                    # acceleration of gravity: meters/second/second
    kp = 0.86*g/W**2            # wavenumber at spectral peak
    cp = np.sqrt(g/kp)          # phase speed at spectral peak
    cm = 0.2305                 # phase speed at gravity/capillary crossover
    km = g/cm**2                # wavenumber at gravity/capillary crossover
    om = W/cp                   # inverse wave age;
    omc = om*np.cos(agetheta)   # directional inverse wave age
    alfp = 6e-3*np.sqrt(om)     # Phillips-Kitaigorodskii parameter 
                                #   (dimensionless)
    sig = 0.08*(1+4/omc**3)     # standard deviation for JONSWAP correction
    ustar = 0.025*W             # friction velocity
    # calculate Elfouhaily spectrum
    # define conditional parameters gam & alfm
    if omc<1:
        gam = 1.7
    else:
        gam = 1.7+6*np.log10(omc)
    if ustar<cm:
        alfm = 1e-2*(1+np.log(ustar/cm))
    else:
        alfm = 1e-2*(1+3*np.log(ustar/cm))
    a0 = np.log(2)/4
    ap = 4
    am = 0.13*(ustar/cm)
    # K-space. Grid is slightly dithered to handle singularities in spectra.
    # This issue can be handled more formally in future release.
    kmax = 2*np.pi/dx
    offset = 0.001*kmax*np.random.rand() # Randomized small grid offsets
    if nyquist=='positive':
        ksamples = (kmax/NFFT)*(np.arange((-NFFT/2+1), (NFFT/2)+1))# - offset
    elif nyquist=='negative':
        ksamples = (kmax/NFFT)*(np.arange((-NFFT/2),(NFFT/2)))# - offset 
    (KX0,KY0) = np.meshgrid(ksamples, ksamples) # function returns these 
    if nyquist=='positive': 
        KX0[NFFT//2-1, NFFT//2-1] = offset
        KY0[NFFT//2-1, NFFT//2-1] = offset                                          #   matrices
    elif nyquist=='negative': 
        KX0[NFFT//2, NFFT//2] = offset
        KY0[NFFT//2, NFFT//2] = offset
    # Rotate K-space. (Future ver: move this rotation out to its own function.)
    KX = KX0*np.cos(windaz) + KY0*np.sin(windaz)
    KY = -KX0*np.sin(windaz) + KY0*np.cos(windaz)
    KXR = KX
    KYR = KY # function also returns these matrices
    k = np.sqrt(KX**2+KY**2)
    # calculate components of spectral model
    LPM = np.exp(-(5/4)*(kp/k)**2)
    GAM = np.exp(-(np.sqrt(k/kp)-1)**2/(2*sig**2))
    Jp = gam**GAM
    F3 = np.exp(-(om/np.sqrt(10))*(np.sqrt(k/kp)-1))
    # low frequency (gravity wave) model
    Blow = 0.5*alfp*np.sqrt(k/kp)*LPM*Jp*F3 
    # high frequency capillary & gravity/capillary wave model
    Bhigh = 0.5*alfm*np.sqrt(k/km)*np.exp(-(1/4)*((k/km)-1)**2)  
    Bhigh = Bhigh*LPM
    # angular variation model
    delk = np.tanh(a0+ap*(kp/k)**(5/4)+am*(k/km)**(5/4))
    modfun = (1/(2*np.pi))*(1+delk*(((KX/k)**2-(KY/k)**2))*np.cos(2*alftemp)+2*
        (KX/k)*(KY/k)*np.sin(2*alftemp))
    # total elevation power spectrum; this matrix is returned to calling 
    #   program
    if cap == True:
        KSPEC = ((Blow+Bhigh)*k**(-4))*(modfun)
    else:
        # decrease high frequency component of spectra
        KSPEC = ((Blow+cap_factor*Bhigh)*k**(-4))*(modfun)
        
    # total elevation amplitude spectrum; this matrix is returned to calling 
    #   program
    if nyquist=='positive':
        KSPEC[NFFT//2-1, NFFT//2-1] = 0.0
        KX0[NFFT//2-1, NFFT//2-1] = 0.0
        KY0[NFFT//2-1, NFFT//2-1] = 0.0
        KXR[NFFT//2-1, NFFT//2-1] = 0.0
        KYR[NFFT//2-1, NFFT//2-1] = 0.0
        k[NFFT//2-1, NFFT//2-1] = 0.0
    elif nyquist=='negative':
        KSPEC[NFFT//2, NFFT//2] = 0.0
        KX0[NFFT//2, NFFT//2] = 0.0
        KY0[NFFT//2, NFFT//2] = 0.0
        KXR[NFFT//2, NFFT//2] = 0.0
        KYR[NFFT//2, NFFT//2] = 0.0
        k[NFFT//2, NFFT//2] = 0.0
    KSPECA = np.sqrt(KSPEC)
    varout = (Blow, Bhigh, k, modfun)
    return KX0, KY0, KXR, KYR, KSPEC, KSPECA, offset, varout

def kspec_to_sse(W, KX, KY, KSPECA):
    '''last revised 25 June 2010, John G Pierce
    this code accepts a real 2-D power spectrum in k-space
    it then calculates phase-randomized inverse transforms
    to produce random realizations of sea surface height, sea surface slope
    and the x and y components of the randomized slope
    '''
    RANPHASE = np.exp(-1j*(2*np.pi)*np.random.rand(*KSPECA.shape))
    dmax = 1 
    RANREP = np.fft.ifft2(RANPHASE*KSPECA)
    SX = np.real(dmax*np.fft.ifft2(np.fft.ifftshift((KX**2)*(KSPECA**2)*
        RANPHASE))) # ECH mod: "k squared method", from Cox-Munk.
    SY = np.real(dmax*np.fft.ifft2(np.fft.ifftshift((KY**2)*(KSPECA**2)*
        RANPHASE)))
    
    # Slope renormalization to Cox-Munk: 
    SX = SX-np.mean(np.mean(SX,0),0)
    SY = SY-np.mean(np.mean(SY,0),0)
    
    # Cox-Munk slopes
    if W>=1:
        sig2g = 0.0103+.0092*np.log(W)
    else:
        sig2g = 0.0103
    sig2w = (1.2e-5)*(W**2.1)*np.log(1+(((2*np.pi)**2)/2.5)*1e4)
    sig2u = (10/19)*sig2g+(2/3)*sig2w
    sig2c = (9/19)*sig2g+(1/3)*sig2w
    SIG2U = np.sqrt(np.mean(SX**2,0)) # ECH. The RMS of the power spec
    SIG2C = np.sqrt(np.mean(SY**2,0))
    RATX = sig2u/SIG2U
    RATY = sig2c/SIG2C
    SX = RATX*SX
    SY = RATY*SY
    return SX, SY, RANREP

def sse_to_imspec(SX, SY, thetar, phir, thetasun, phisun):
    '''last revised 29 June 2010, John G. Pierce
    '''
    d2rad = np.pi/180
    N = len(SX)
    cosmuthresh = np.cos(0.009) # cosine of angular size of sun, to keep 
        # radiance formula bounded
    Isat = 1000 # saturation value (for extreme glint)
    # sky scattering constant
    ksky = 0
    I0 = 10
    n = 1.34 # refractive index seawater 1.34-1.35
    n2 = n**2
    # coordinates of receiver -- theta here is elevation angle, not polar angle.
    thetar = d2rad*thetar
    phir = d2rad*phir
    urec = np.array([
             np.cos(thetar)*np.sin(phir),
             np.cos(thetar)*np.cos(phir),
             np.sin(thetar)
             ])
    urec = urec/np.linalg.norm(urec)
    # coordinates of sun -- theta here is elev angle, not polar. 
    thetasun = d2rad*thetasun
    phisun = d2rad*phisun
    usun = np.array([
                [np.cos(thetasun)*np.sin(phisun)], 
                [np.cos(thetasun)*np.cos(phisun)], 
                [np.sin(thetasun)]
                ])
    usun = usun/np.linalg.norm(usun)
    # unit normal to surface facet
    s2 = SX**2 + SY**2;
    DENI = 1/(np.sqrt(1+s2)) # z-coordinate of unit normal to surface facet
    nu = np.zeros([N,N,3]) 
    nu[:,:,0] = -SX*DENI 
    nu[:,:,1] = -SY*DENI
    nu[:,:,2] = DENI
    # 
    Ip = np.zeros([N, N])
    Is = np.zeros([N, N])
    I  = np.zeros([N, N])
    rs = np.zeros([N, N])
    rp = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            uss = np.squeeze(nu[i, j, :])
            cosbeta = np.dot(uss.T, urec)
            beta = np.arccos(cosbeta) # angle between surface normal and 
                # receiver direction
            sinbeta = np.sin(beta)
            sinbeta2 = sinbeta**2
            rs[i,j] = ( cosbeta - np.sqrt(n2-sinbeta2) ) / ( cosbeta + 
                np.sqrt(n2-sinbeta2) )
            rp[i,j] = ( n2*cosbeta - np.sqrt(n2-sinbeta2) ) / ( n2*cosbeta + 
                np.sqrt(n2-sinbeta2) )
            usky = 2*np.dot(urec.T, uss)*uss - urec # unit vector pointing from 
                # facet toward specular source point in sky
            costhetasky = np.dot(np.array([[0],[0],[1]]).T, usky)
            expfactor = 1.0 - np.exp(-(0.32)/costhetasky)
            cosmu = np.dot(usky.T,usun)
            if cosmu<cosmuthresh:
                Ip[i,j]= (I0/2) * ( (cosmu**2)/(1-cosmu) + ksky ) * expfactor
                Is[i,j]= (I0/2) * ( 1/(1-cosmu) + ksky ) * expfactor
            else:
                Ip[i,j] = Isat
                Is[i,j] = Isat
    RHH = rs**2 #(np.abs(rs))**2
    RVV = rp**2 #(np.abs(rp))**2
    ImageVV = RVV*Ip;
    ImageHH = RHH*Is;
    I = ImageVV+ImageHH; # total image
    IMSPECtot = np.fft.fftshift(np.fft.fft2(I-np.mean(I))); # image fft
    return I, IMSPECtot
