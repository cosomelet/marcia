from numba import jit
import numpy as np
from scipy.integrate import odeint


# To combine the above functions into a single function, we need to use the following:
@jit(nopython=True)
def gMdef(tau, l1, nu):
    # This for example is the 7/2 case, now we have only the generalised funciton !!
    # result =  np.sqrt(2./5.) * 7**(3./4.) * np.exp(- np.sqrt(7.) * np.abs(tau) / l1) * (1./l1**2.)**(1./4.) * np.abs(tau) * (1. + l1 / np.sqrt(7. * tau**2.))/ l1

    A = (nu / (2. * np.pi * l1**2.))**(1./8.) 
    #* (sp.special.gamma(nu/2. - 1./4.)/sp.special.gamma(nu/2. + 1./4.))**(1./2.) * (sp.special.gamma(nu + 1./2.)/sp.special.gamma(nu))**(1./4.)
    B = 2.**(5./4. - nu/2.) 
    # / sp.special.gamma(nu/2. - 1./4.)
    C = (2. * nu)**(1./2.) * tau**2. / l1
    result = A,B,C # A**2. * B * C**(nu/2. - 1./4.) * sp.special.kv(nu/2. - 1./4., C)
    return result

@jit(nopython=True)
def gdMdef(tau, l1, nu):
    x = np.abs(tau)
    l_s = l1 # just renaming
    A = 2**(11./8. - nu/4.) * (nu / l_s**2.)**(1./4.) * (np.sqrt(nu) * np.sqrt(x**2.) / l_s)**(3./4. + nu/2.)
    D = np.pi**(1./4.) * x 
    # * sp.special.gamma(1./4. + nu/2.)
    return A, D 

    