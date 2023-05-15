from numba import jit
import numpy as np
from scipy.integrate import odeint

@jit(nopython=True)
def hubble_rate(H0,Omega_m,Omega_b,Omega_k,de, z):
    Omega_r = 4.18343*10**-5./(H0/100.)**2.
    E2 = Omega_r*((1. + z)**4) + Omega_m*((1. + z)**3) + Omega_b*((1. + z)**3) + Omega_k*((1. + z)**2) + (1. - Omega_m - Omega_k - Omega_b - Omega_r)*de
    Hofz = H0*np.sqrt(E2)
    return Hofz

@jit(nopython=True)
def dark_energy_f_wCDM(w0,wa,z):
    return np.exp(3.*(-wa + wa/(1. + z) + (1. + w0 + wa)*np.log(1. + z)))

@jit(nopython=True)
def inv_hubble_rate(H0,Omega_m,Omega_b,Omega_k,de, z):
    return 1./hubble_rate(H0,Omega_m,Omega_b,Omega_k,de, z)

@jit(nopython=True)    
def sound_horizon(H0,Omega_b,Omega_m,Obsample):
    m_nu = 0.06 # In the units of eV
    omega_nu = 0.0107 *(m_nu/1.0) #This is in the units of eV. This should be equal to 6.42*10^(-4)
    if Obsample:
        omega_b = Omega_b*(H0/100.)**2. #0.0217
    else:
        omega_b = 0.0217
            
    omega_cb = (Omega_m+Omega_b)*(H0/100.)**2 - omega_nu
    if omega_cb < 0:
        rd = -1.0
    else:
        rd = 55.154 * np.exp(-72.3* ((omega_nu + 0.0006)**2))/((omega_cb**0.25351)*(omega_b**0.12807))
        if np.isnan(rd):
            rd = 0.0
    return rd

@jit(nopython=True)
def z_inp(z):
    return np.arange(0.,np.max(z)+.5,0.01)

@jit(nopython=True)
def interpolate(z_inp,z,func):
    return np.interp(z,z_inp,func)
   