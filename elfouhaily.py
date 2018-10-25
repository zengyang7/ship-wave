
import numpy as np
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors, ticker
from scipy.interpolate import interp2d

## Original functions from Dr. Paterson.
from elfouhaily_org import sse_kspec, kspec_to_sse, sse_to_imspec

## Wrappers, fixes, and add-ons for functions in 'elfouhaily_org.py'
def elfouhaily_varspec(W,windaz, agetheta, dx, NFFT, alftemp, cap=True, 
    cap_factor = 0.0):
    '''Wrapper: calculates KX, KY, KSPEC from 'elfouhaily_org.py'.
        
        KSPEC: Elfouhaily 1-sided, continuous, directional variance spectrum.
    '''
    KX, KY, _, _, KSPEC, _, offset, varout = \
        sse_kspec(W, windaz, agetheta, dx, NFFT, alftemp, 
            cap=cap, cap_factor=cap_factor, nyquist='negative')
    return KX, KY, KSPEC, offset, varout

def cont1sVar_to_disc2sAmp(dk, KSPEC, debug=0, offset=0, hermitian_option=0, 
    seed_flag=False, seed=0):
    '''Creates a discrete, 2-sided amplitude spectrum from the continuous, 
        1-sided variance spectrum.

        KSPEC: Elfouhaily 1-sided, continuous, directional variance spectrum.
        KSPEC0: 2-sided, discrete variance spectrum.
        S_dft0: 2-sided, discrete amplitude spectrum with no phase information.
        S_dft: 2-sided, discrete amplitude spectrum including random phases, 
            with Hermitian symmetry.
    '''
    N = KSPEC.shape[0]
    KSPEC0 = 0.5*dk*dk * KSPEC
    KSPEC0[0,0] *= 2.
    KSPEC0[0,N//2] *= 2.
    KSPEC0[N//2,0] *= 2.
    KSPEC0[N//2,N//2] *= 2.
    S_dft0 = np.sqrt(KSPEC0)
    S_dft = make_hermitian_ranphase(S_dft0.astype(complex), 
        option=hermitian_option, seed_flag=seed_flag, seed=seed)
    # debug
    if debug:
        _debug_c1v_to_d2a(debug, KSPEC, S_dft0, S_dft, dk, offset)
    return S_dft

def make_hermitian_ranphase(S_dft0, option=0, seed_flag=False, seed=0):
    '''Add random phase and enforce Hermitian symmetry.

        S_dft0: 2-sided, discrete, amplitude spectrum (sqrt(0.5*dkx*dky*KSPEC))
        option:
            0 - Random phase and magnitude. r = a + ib with a,b standard normal 
                distributions N(0,1).
            1 - Random phase and magnitude. r = e^(2*pi*a) with 'a' an uniform 
                distribution U[0,1].
            2 - Random phase only. r as in (2) but with Hermitian symmetry. 
                Requires the input S_dft0 to be centro-symmetric. 
    '''
    N = S_dft0.shape[0]
    if S_dft0.shape[1]!=N or len(S_dft0.shape)>2:
        raise IndexError('S_dft0 should be a 2D square array.')
    if seed_flag:
        np.random.seed(seed)  # same random seed for the realizations
    # Create random matrix and scaling factor.
    if option==0:
        f = 1/np.sqrt(2)
        a = np.random.normal(0., 1., (N, N))
        b = np.random.normal(0., 1., (N, N))
        r = (1/np.sqrt(2)) * (a + 1j*b)
        r[0,0] = r[N//2,0] = r[0,N//2] = r[N//2,N//2] = 1. + 0j        
    elif option==1:
        f = 1/np.sqrt(2)
        r = np.exp(1j*2.*np.pi*np.random.rand(N,N))
        r[0,0] = r[N//2,0] = r[0,N//2] = r[N//2,N//2] = 1. + 0j
    elif option==2:
        f = 0.5
        r = np.ones([N,N],dtype=complex)
        # center
        r[1:N, 1:N//2] = np.exp(1j*2.*np.pi*np.random.rand(N-1, N//2-1))
        r[1:N, N//2+1:N] = np.conj(np.flipud(np.fliplr(r[1:N,1:N//2])))
        # left column
        r[1:N//2, 0] = np.exp(1j*2.*np.pi*np.random.rand(N//2-1))
        r[N//2+1:N, 0] = np.conj(r[1:N//2, 0][::-1])
        # middle column
        r[1:N//2, N//2] = np.exp(1j*2.*np.pi*np.random.rand(N//2-1))
        r[N//2+1:N, N//2] = np.conj(r[1:N//2, N//2][::-1])
        # top row
        r[0, 1:N//2] = np.exp(1j*2.*np.pi*np.random.rand(N//2-1))
        r[0, N//2+1:N] = np.conj(r[0, 1:N//2][::-1])
    else:
        f = 1
        r = np.ones([N,N],dtype=complex)
    S_dft = S_dft0*r 
    # Make hermitian
    # center
    a = S_dft[1:N, 1:N//2]
    b = np.flipud(np.fliplr(S_dft[1:N, N//2+1:]))
    S_dft[1:N, 1:N//2] = f * (a + np.conj(b))
    S_dft[1:N, N//2+1:] = np.conj(np.flipud(np.fliplr(S_dft[1:N, 1:N//2])))
    # left column
    a = S_dft[1:N//2, 0]
    b = S_dft[N//2+1:, 0][::-1]
    S_dft[1:N//2, 0] = f * (a + np.conj(b))
    S_dft[N//2+1:, 0] = np.conj(S_dft[1:N//2, 0])[::-1]
    # middle column
    a = S_dft[1:N//2, N//2]
    b = S_dft[N//2+1:, N//2][::-1]
    S_dft[1:N//2, N//2] = f * (a + np.conj(b))
    S_dft[N//2+1:, N//2] = np.conj(S_dft[1:N//2, N//2])[::-1]
    # top row
    a = S_dft[0, 1:N//2]
    b = S_dft[0, N//2+1:][::-1]
    S_dft[0, 1:N//2] = f * (a + np.conj(b))
    S_dft[0, N//2+1:] = np.conj(S_dft[0, 1:N//2]) [::-1]  
    return S_dft

def elev_to_slope(KX, KY, S_dft, offset=0.0):
    '''Get x and y slopes from amplitude spectrum (S_dft).
    '''
    SX = S_dft * 1j*(KX + offset)
    SY = S_dft * 1j*(KY + offset)
    SX[:,0] *= 0
    SY[0,:] *= 0
    return SX, SY

def spec_to_sse(S_dft, debug=0):
    '''Creates sea state elevation from a discrete, 2-sided amplitude spectrum. 
    Can also be used to create the slope, from the slope spectrum.
    '''
    N = S_dft.shape[0]
    S_dft_ifft = S_dft * N**2. # NumPy scaling
    S_dft_ifft = shift_spec(S_dft_ifft) # shift order  
    SSE = np.fft.ifft2(S_dft_ifft)
    #debug
    if debug:
        _debug_spec_to_sse(SSE, S_dft_ifft)
    return np.real(SSE)


## Utils
def roll_nyquist(KX0, KY0, KSPEC, offset=0):
    '''Moves the Nyquist frequencies from positive to negative.
    '''
    KX = np.roll(KX0, 1, 1)
    KX[:,0] *= -1
    KX[:,0] -= 2*offset
    KY = np.roll(KY0, 1, 0)
    KY[0,:] *= -1
    KY[0,:] -= 2*offset
    KSPEC = np.roll(KSPEC, 1, 0)
    KSPEC = np.roll(KSPEC, 1, 1)
    return KX, KY, KSPEC

def shift_spec(SPEC):
    '''Shifts the spectrum to/from FFT order. Also works for KX and KY.
    '''
    return np.fft.ifftshift(SPEC)

def freq_to_coords(N, dk):
    '''Create physical coordinates from frequency information.
    '''
    L = 2.*np.pi/dk
    dx = L/N
    x = np.arange(N)*dx
    X,Y = np.meshgrid(x, x)
    return X, Y

def coords_to_freq(N, dx, offsest=0):
    ''' Creates frequencies from physical coordinates.
    '''
    L = dx * N
    dk = 2.*np.pi/L 
    KX, KY = make_K(dk, N, offset)
    return KX, KY

def make_K(dk, N, offset=0):
    '''Creates the frequency matrices.
    '''
    k = np.arange(-1.*dk*(N/2.)-offset, (N/2.)*dk-offset, dk)
    KX, KY = np.meshgrid(k, k)
    return KX, KY

def rad_to_slope(rad):
    ''' Converts angle in radians to slope (dy/dx).
    '''
    return np.tan(rad)

def slope_to_rad(slope):
    ''' Converts slope (dy/dx) to angle in radians.
    '''
    return np.arctan(slope)

def angle(c):
    '''Returns the angle of a complex number. Returns angle of zero for 
    numbers with zero magnitude
    '''
    a = np.logical_not(np.isclose(0,np.abs(c)))
    return np.angle(c)*a

def calc_slope(dx, SSE):
    ''' Calculates (numerical) the slope of a 2D surface.
    '''
    N = SSE.shape[0]
    dy = dx
    slope_y, slope_x = np.gradient(SSE, dy, dx)
    slope_x = np.arctan(slope_x)
    slope_y = np.arctan(slope_y)
    return slope_x, slope_y


## Plots
#def plot_surf(X, Y, ZZ, x_label='x', y_label='y', z_label='z', title='', 
#    bar=True, stride=1, cmap=plt.cm.jet, grid=True, interpolation='none',
#    contour=False, levels=[], log=False):
#    ''' bar: whether to plot 3D as bars or surface.
#        stride: For surface plot (bar=False) specify plotting stride. 
#                1 to plot all, >1 to plot faster.
#        interpolation: e.g. 'none', 'bilinear', etc.
#        contour: whether the 2D plots are contours.
#        levels: for contours.
#        log: whether contour is in log scale.
#    '''
#    if type(ZZ)==np.ndarray:
#        ZZ = [ZZ]
#    N = len(ZZ)
#    fig = plt.figure(figsize=(10,5*N))
#
#    for i,Z in enumerate(ZZ):
#        dx, dy = X[0,1]-X[0,0], Y[1,0]-Y[0,0]
#        # subplot 1
#        ax1 = fig.add_subplot(N, 2, 2*i+1, projection='3d')
#        if bar:
#            x, y = X.ravel(), Y.ravel()
#            top = Z.ravel()
#            bottom = np.zeros_like(top)
#            fracs = top/top.max()
#            norm = colors.Normalize(fracs.min(), fracs.max())
#            color = cmap(norm(fracs))
#            ax1.bar3d(x, y, bottom, dx, dy, top, color=color)
#        else:
#            if i == 0:
#                global_min = np.min(Z)
#                global_max = np.max(Z)
#            ax1.plot_surface(X, Y, Z, rstride=stride, cstride=stride, cmap=cmap)
#        ax1.set_xlabel(x_label)
#        ax1.set_ylabel(y_label)
#        ax1.set_zlabel(z_label)
#        ax1.set_zlim(global_min, global_max)
#        # subplot 2
#        ax2 = fig.add_subplot(N, 2, 2*i+2)
#        extent = (X[0,0]-dx/2., X[0,-1]+dx/2., Y[-1,0]+dy/2., Y[0,0]-dy/2.)
#        if contour:
#            if log:
#                pcol = plt.contourf(X, Y, Z, levels, cmap=cmap, 
#                    locator=ticker.LogLocator())#, extend='both')
#            else:
#                pcol = plt.contourf(X, Y, Z, levels, cmap=cmap, extend='both')
#            pcol.cmap.set_under('black')
#            pcol.cmap.set_over('white')
#        else:
#            pcol = ax2.imshow(Z, cmap=cmap, extent=extent, interpolation=interpolation, vmin=global_min, vmax=global_max)
#        ax2.set_xlabel(x_label)
#        ax2.set_ylabel(y_label)
#        if grid:
#            ax2.set_xticks(np.arange(extent[0], extent[1], dx), minor=True);
#            ax2.set_yticks(np.arange(extent[2], extent[3], -dy), minor=True);
#            ax2.grid(which='minor', color='w', linestyle='-', linewidth=1)
#        # fig
#        fig.colorbar(pcol)
#    plt.suptitle(title)
#    plt.tight_layout(pad=3.0)
#    return fig


# Checks & debug
def _print_check(C):
    if C:
        print('PASS!')
    else:
        print('FAIL!')
    return

def check_parsevals(f, F, N, rtol=1e-05, atol=1e-08, equal_nan=False, 
    print=True):
    a = np.sum(np.abs(f)**2.)
    b = np.sum(np.abs(F)**2.)/(N*N)
    C = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if print:
        _print_check(C)
    return C

def check_elev(elev, rtol=1e-05, atol=1e-08, equal_nan=False, print=True):
    '''Checks if all values are real.
    '''
    a = np.sum(np.abs(np.imag(elev)))
    b = 0.0
    C = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if print:
        _print_check(C)
    return C

def check_hermitian(M, rtol=1e-05, atol=1e-08, equal_nan=False, print=True):
    a = np.matrix(M[1:,1:])
    b = np.conj(np.fliplr(np.flipud(a)))
    C1 = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    a = np.matrix(M[1:,0])
    b = np.conj(np.fliplr(np.flipud(a)))
    C2 = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    a = np.matrix(M[0,1:])
    b = np.conj(np.fliplr(np.flipud(a)))
    C3 = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    C = C1 and C2 and C3
    if print:
        _print_check(C)
    return C

def check_dc(M, rtol=1e-05, atol=1e-08, equal_nan=False, print=True):
    C = np.isclose(M[0,0], 0.+0.j, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if print:
        _print_check(C)
    return C

def check_spreading_function(KX, KY, spread, Ntheta=1000, NK=10):
    N = KX.shape[0]
    func = interp2d(KX, KY, spread)
    theta = np.linspace(0, 2*np.pi, Ntheta)
    KX[N//2-1,:N//2][::-1]*-1
    for k in np.linspace(np.abs(KX[N//2-1,N//2-1]), KX[N//2-1,-1], NK):
        kx = k*np.cos(theta)
        ky = k*np.sin(theta)
        dtheta = theta[1]-theta[0]
        f = 0
        for i in range(Ntheta):
            f += func(kx[i],ky[i])
        F = f*dtheta
        print('  K = {}    integral = {}'.format(k, F))

def _debug_c1v_to_d2a(debug, KSPEC, S_dft0, S_dft, dk, offset):
    N = S_dft0.shape[0]
    print('KSPEC Hermitian? (should fail):')
    _ = check_hermitian(KSPEC)
    print('S_dft0 Hermitian? (should fail):')
    _ = check_hermitian(S_dft0)
    print('S_dft Hermitian? (should pass):')
    _ = check_hermitian(S_dft)
    print('\nTotal Variance KSPEC: {}'.format(np.sum(KSPEC*0.5*dk*dk)))
    print('Total Variance S_dft0:  {}'.format(np.sum(S_dft0**2)))
    print('Total Variance S_dft:   {}'.format(np.sum(np.abs((S_dft/N**2)**2))))
    print('\nMax Variance, 2-sided discrete: {}'.format(
                                                np.max(KSPEC*0.5*dk*dk)))
    print('Max Variance, 1-sided continuous: {}'.format(np.max(KSPEC)))
    if debug>1:
        KX, KY = make_K(dk, N, offset)
        bar = False
        stride = 1
        grid = False
        interpolation = 'none'
        
        # change by zeng
        #cmap = plt.cm.jet
        #_ = plot_surf(KX, KY, KSPEC, cmap=cmap,
        #        x_label='Kx', y_label='Ky', z_label='$S_{Elf}$', title='', 
        #        bar=bar, stride=stride, grid=grid, interpolation=interpolation)
        #_ = plot_surf(KX, KY, np.abs(S_dft0)**2/(0.5*dk*dk), cmap=cmap,
        #        x_label='Kx', y_label='Ky', z_label='$S_{Elf}$', title='', 
        #        bar=bar, stride=stride, grid=grid, interpolation=interpolation)
        #_ = plot_surf(KX, KY, np.abs(S_dft/N**2)**2./(0.5*dk*dk), cmap=cmap,
        #        x_label='Kx', y_label='Ky', z_label='$S_{Elf}$', title='', 
        #        bar=bar, stride=stride, grid=grid, interpolation=interpolation)
        #plt.show()
    return

def _debug_spec_to_sse(SSE, S_dft_s):
    N = S_dft_s.shape[0]
    print('\nElevation:')
    print('S_dft shifted, Hermitian? DC=0?')
    _ = check_hermitian(S_dft_s)
    _ = check_dc(S_dft_s)
    print('Parsevals? Elev real?')
    check_parsevals(SSE, S_dft_s, N)
    check_elev(SSE)
    print('Mean: {}'.format(np.mean(np.real(SSE))))
    print('Max: {}'.format(np.max(np.real(SSE))))
    print('Min: {}'.format(np.min(np.real(SSE))))
    return 

