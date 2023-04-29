import numpy as np
import scipy.integrate
import scipy.integrate as it
from scipy.integrate import quad
import scipy.constants as const
import sys
import time
import cmath
import os
from marcia.database import Data
from marcia.params import Params

class Cosmology(object):
    """
        General background theory that all the theories must be inherited from.
        Inputs would be the appropriate cosmological parameters needed for the particular model. 
        To change
        """
    
    def __init__(self, params,prior_file=None):
        # model = [] with rd, mb ,ob
        # params = ['ho'] 
        self.param = Params(params,prior_file)
        self.priors = self.param.Priors
        self.labels = self.param.Labels


        self.rdsample = False
        self.Mbsample = False
        self.Obsample = False
        

        if 'r_d' in params:
            self.rdsample = True
        if 'M_b' in params:
            self.Mbsample = True
        if 'Omega_b' in params:
            self.Obsample = True


    
        if self.rdsample and self.Obsample:
            raise ValueError('Mb and Ob cannot be sampled together')
        

        self.clight = 299792458. / 1000.
    
    @staticmethod
    def a(z):
        return 1. / (1. + z)

    def dark_energy_f(self,parameters, z):
        pass

    def dark_energy_w(self,parameters, z):
        pass
    
    def hubble_rate(self,parameters, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        p = self.param(parameters)
        zeq = 2.5 * 10.**4. * (p.Omega_m + p.Omega_b) * (p.H0/100.)**2. /(2.7255/2.7)**4.
        Omega_r = 4.18343*10**-5./(p.H0/100.)**2.
        E2 = Omega_r*pow((1. + z),4.) + p.Omega_m*pow((1. + z),3.) + p.Omega_b*pow((1. + z),3.) + p.Omega_k*pow((1. + z),2.) + (1. - p.Omega_m - p.Omega_k - p.Omega_b - Omega_r)*self.dark_energy_f(parameters,z)
        Hofz = p.H0*np.sqrt( E2 )
        return np.nan_to_num(Hofz)

    def transverse_distance(self,parameters, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        def Ly(y,t):
            # Here this could be replaced with expansion_rate_int if needed 
            return 1./self.hubble_rate(parameters,t)

        y=it.odeint(Ly,0.0,z)
        tint = np.array(y[:,0])
        return np.nan_to_num( self.clight* tint)
       

    #To define the value of r_d using the Aubourg15 formulae
    def sound_horizon(self,parameters):
        p = self.param(parameters)
        if self.rdsample:
            rd = p.r_d
        else:
            m_nu = 0.06 # In the units of eV
            omega_nu = 0.0107 *(m_nu/1.0) #This is in the units of eV. This should be equal to 6.42*10^(-4)
            if self.Obsample:
                omega_b = p.Omega_b*(p.H0/100.)**2. #0.0217
            else:
                omega_b = 0.0217
            
            omega_cb = (p.Omega_m+p.Omega_b)*(p.H0/100.)**2 - omega_nu
            if omega_cb < 0:
                rd = -1.0
            else:
                rd = np.nan_to_num(55.154 * np.exp(-72.3* pow(omega_nu + 0.0006,2))/(pow(omega_cb, 0.25351)*pow(omega_b, 0.12807)))

        return rd

    # This is to model the mu, but however the Mb is set inside, therefore it is M_b + mu but not mu alone
    def distance_modulus(self,parameters,z1, z2):
        p = self.param(parameters)
        if self.Mbsample:
            Mb = p.M_b
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(z1) )
        else:
            # This is useful if only SN data is needed to be used
            Mb = -19.05
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(z1) )
        return mu
    
    # This is to provide the Mb alone
    def Abs_M(self,parameters):
        p = self.param(parameters)
        if self.Mbsample:
            Mb = p.M_b
        else:
            # This is useful if only SN data is needed to be used
            Mb = -19.05
        return Mb

    def z_rec(self,parameters):
        '''
        redshift of recombination, from Hu & Sugiyama 1996 fitting formula
        '''
        p = self.param(parameters)
        
        omega_b = p.Omega_b*(p.H0/100.)**2.
        omega_m = p.Omega_m*(p.H0/100.)**2. + omega_b
        g1 = 0.0738*omega_b**(-0.238)/(1.+39.5*omega_b**0.763)
        g2 = 0.560/(1.+21.1*omega_b**1.81)
        return 1048.*(1.+0.00124*omega_b**(-0.738))*(1.+g1*omega_m**g2)

    def rs(self,parameters,z):
        '''
        sound horizon at z, the variable is x = ln(a)
        '''
        p = self.param(parameters)
        omega_b = p.Omega_b*(p.H0/100.)**2.
        Rs = 31500./(2.7255/2.7)**4. #3./4./2.47282 *1e+05   # just photons are coupled to baryons
        return self.clight/p.H0 * quad(func=lambda x: np.exp(-x)/(self.expansion_rate(np.exp(-x)-1.)*np.sqrt(3.*(1. + Rs * omega_b * np.exp(x) )) ), a = np.log(1e-40) , b = np.log(1./(1.+z)) , limit=100 )[0]


    def rs2(self,parameters,z):
        '''
        sound horizon at z, the variable is x = a
        '''
        p = self.param(parameters)
        omega_b = p.Omega_b*(p.H0/100.)**2.
        Rs = 31500./(2.7255/2.7)**4. #3./4./2.47282 *1e+05   # just photons are coupled to baryons
        return self.clight/p.H0 * quad(func=lambda x: 1./(x**2. * self.expansion_rate(1./x - 1.)*np.sqrt(3.*(1. + Rs * omega_b * x )) ), a = 0 , b = 1./(1. + z))[0]
    

class wCDM(Cosmology):
    def __init__(self, params,prior_file=None):
        super().__init__(params,prior_file)
        
    
    def dark_energy_f(self, parameters, z):
        p = self.param(parameters)
        return np.exp(3.*(-p.wa + p.wa/(1. + z) + (1. + p.w0 + p.wa)*np.log(1. + z)))
    
    def dark_energy_w(self,parameters, z):
        p = self.param(parameters)
        a = self.a(z)
        return p.w0 + p.wa*(1-a)
    
class LCDM(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)

# class kLCDM(wCDM,rdsample=False,Obsample=False,Mbsample=False):
#     def __init__(self, parameters,rdsample,Obsample,Mbsample):
#         super().__init__(parameters)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = parameters[2]
#         self.w_0 = -1.0
#         self.w_a = 0.0
#         self.params = ["H0", "Omega_m", "Omega_k"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-0.7,0.6]]

# class kwCDM(wCDM):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = parameters[2]
#         self.w_0 = parameters[3]
#         self.w_a = 0.0
#         self.params = ["H0", "Omega_m", "Omega_k", "w"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$w$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-0.4,0.2],[-1.2,-0.5]]

# class CPL(wCDM):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = 0.0
#         self.w_0 = parameters[2]
#         self.w_a = parameters[3]
#         self.params = ["H0", "Omega_m", "w0", "wa"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$w_0$", "$w_a$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-2.5,0.5],[-10.0,10.0]]

# class kCPL(wCDM):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = parameters[2]
#         self.w_0 = parameters[3]
#         self.w_a = parameters[4]
#         self.params = ["H0", "Omega_m", "Omega_k", "w0", "wa"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-0.5,0.4],[-2.5,0.5],[-10.0,10.0]]
#         self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$w_0$", "$w_a$"]


# class ThreeCPL(Cosmology):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample) 
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = 0.0
#         self.w_0 = parameters[2]
#         self.w_a = parameters[3]
#         self.w_b = 2.*parameters[4]
#         self.w_c = 6.*parameters[5]
#         self.w_d = 24.*0.0
#         self.params = ["H0", "Omega_m", "w0", "wa", "wb", "wc"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$w_0$", "$w_a$", "$w_b$", "$w_c$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-3.0,0.5],[-30.,20.],[-50.,120.],[-150.,80.]] 

#     def dark_energy_f(self, z):
#         a = self.a(z)
#         intwz = -3.*((self.w_a + self.w_b/2. +self.w_c/6. + self.w_d/24.)*pow(a,3.)*pow(1.-a,1.) +
#                             (3.*self.w_a + 7.*self.w_b/4. + 7.*self.w_c/12. + 7.*self.w_d/48)*pow(a,2.)*pow(1.-a,2.) +
#                             (3.*self.w_a + 2.*self.w_b + 13.*self.w_c/18. + 13.*self.w_d/72.)*pow(a,1.)*pow(1.-a,3.) +
#                             (self.w_a + 3.*self.w_b/4. + 11.*self.w_c/36. + 25.*self.w_d/288.)*pow(1.-a,4.) +
#                             (1. + self.w_0 + self.w_a + self.w_b/2. + self.w_c/6. + self.w_d/24.)*np.log(a))
#         if intwz > 10.0:
#             return np.exp(10)
#         else:
#             return np.exp(intwz)
        
#     def dark_energy_w(self, z):
#         a = self.a(z)
#         return self.w_0 + self.w_a*(1.-a) + self.w_b*pow((1.-a),2.)/2. + self.w_c*pow((1.-a),3.)/6. + self.w_d * pow((1.-a),4.)/24.
        
        
# class k3CPL(ThreeCPL):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = parameters[2]
#         self.w_0 = parameters[3]
#         self.w_a = parameters[4]
#         self.w_b = 2.*parameters[5]
#         self.w_c = 6.*parameters[6]
#         self.w_d = 24.*0.0
#         self.params = ["H0", "Omega_m", "Omega_k", "w0", "wa", "wb", "wc"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$w_0$", "$w_a$", "$w_b$", "$w_c$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-0.5,0.4],[-3.0,0.5],[-50.,40.],[-100.,200.],[-200.,100.]]

# class XCDM(Cosmology):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = 0.0
#         self.f_0 = 1.0 #parameters[2]
#         self.f_a = parameters[2]
#         self.f_b = parameters[3]
#         self.f_c = parameters[4]
#         self.params = ["H0", "Omega_m", "fa", "fb", "fc"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$f_a$", "$f_b$", "$f_c$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-20.,20.],[-50.,50.],[-50.,50.]]
    
#     def dark_energy_f(self, z):
#         a = self.a(z)
#         return self.f_0 + self.f_a*pow((1.-a),1.) + self.f_b*pow((1.-a),2.) + self.f_c*pow((1.-a),3.)


#     def dark_energy_w(self, z):
#         a = self.a(z)
#         return -1. + ( self.f_a * pow(1. + z,2) + z * (3.* self.f_c * z + 2. * self.f_b * (1. + z)) ) / ( 3. * (self.f_0 * pow(1. + z,3) + z * ( self.f_a * pow(1. + z,2) + z * ( self.f_b * (1. + z) + z * self.f_c ) ) ))

    
# class kXCDM(XCDM):
#     def __init__(self, parameters,rdsample=False,Obsample=False,Mbsample=False):
#         super().__init__(parameters,rdsample,Obsample,Mbsample)
#         self.H_0 = parameters[0]
#         self.Omega_m = parameters[1]
#         self.Omega_b = 0.0
#         self.Omega_k = parameters[2]
#         self.f_0 = 1.0 #parameters[2]
#         self.f_a = parameters[3]
#         self.f_b = parameters[4]
#         self.f_c = parameters[5]
#         self.params = ["H0", "Omega_m", "Omega_k", "fa", "fb", "fc"]
#         self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$f_a$", "$f_b$", "$f_c$"]
#         self.priors = [[61.0,76.0],[0.1,0.5],[-0.5,0.4],[-20.,20.],[-50.,50.],[-50.,50.]]

