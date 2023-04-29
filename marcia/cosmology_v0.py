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

class Cosmology(object):
    """
        General background theory that all the theories must be inherited from.
        Inputs would be the appropriate cosmological parameters needed for the particular model. 
        To change
        """
    
    def __init__(self, model, parameters):
        #self.zlist = z
        # To add the expected nuisance parameters of the model here: model = ['LCDM', 'rd']
        self.model = model[0]
        if len(model) > 1:
            self.nparams = len(parameters) - (len(model) -1)
        else:
            self.rdsample = 'False'
            self.Mbsample = 'False'
            self.Obsample = 'False'
            self.nparams = len(parameters)
        

        self.clight = 299792458. / 1000.
        # Maybe here I would like to add a first check for verifying number of parameters.
        # The order of the parameters is usually like H0, Omega_m, Omega_k, w-parameters
        if self.model == 'LCDM' and self.nparams == 2:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = 0.0
            self.w_0 = -1.0
            self.w_a = 0.0
            self.params = ["H0", "Omega_m"]
            self.labels = ["$H_0$", "$\Omega_m$"]
            self.priors = [[61.0,76.0],[0.1,0.5]]
        elif self.model == 'wCDM' and self.nparams == 3:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = 0.0
            self.w_0 = parameters[2]
            self.w_a = 0.0
            self.params = ["H0", "Omega_m", "w"]
            self.labels = ["$H_0$", "$\Omega_m$", "$w$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-2.5,0.5]]
        elif self.model == 'kLCDM' and self.nparams == 3:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = parameters[2]
            self.w_0 = -1.0
            self.w_a = 0.0
            self.params = ["H0", "Omega_m", "Omega_k"]
            self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-0.7,0.6]]
        elif self.model == 'kwCDM' and self.nparams == 4:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = parameters[2]
            self.w_0 = parameters[3]
            self.w_a = 0.0
            self.params = ["H0", "Omega_m", "Omega_k", "w"]
            self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$w$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-0.4,0.2],[-1.2,-0.5]]
        elif self.model == 'CPL' and self.nparams == 4:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = 0.0
            self.w_0 = parameters[2]
            self.w_a = parameters[3]
            self.params = ["H0", "Omega_m", "w0", "wa"]
            self.labels = ["$H_0$", "$\Omega_m$", "$w_0$", "$w_a$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-2.5,0.5],[-10.0,10.0]]
        elif self.model == 'kCPL' and self.nparams == 5:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = parameters[2]
            self.w_0 = parameters[3]
            self.w_a = parameters[4]
            self.params = ["H0", "Omega_m", "Omega_k", "w0", "wa"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-0.5,0.4],[-2.5,0.5],[-10.0,10.0]]
            self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$w_0$", "$w_a$"]
        elif self.model == '3CPL' and self.nparams == 6:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = 0.0
            self.w_0 = parameters[2]
            self.w_a = parameters[3]
            self.w_b = 2.*parameters[4]
            self.w_c = 6.*parameters[5]
            self.w_d = 24.*0.0
            self.params = ["H0", "Omega_m", "w0", "wa", "wb", "wc"]
            self.labels = ["$H_0$", "$\Omega_m$", "$w_0$", "$w_a$", "$w_b$", "$w_c$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-3.0,0.5],[-30.,20.],[-50.,120.],[-150.,80.]] #[[61.0,76.0],[0.2,0.4],[-5.0,0.5],[-200.,200.],[-200.,200.],[-200.,200.]]
        elif self.model == 'k3CPL' and self.nparams == 7:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = parameters[2]
            self.w_0 = parameters[3]
            self.w_a = parameters[4]
            self.w_b = 2.*parameters[5]
            self.w_c = 6.*parameters[6]
            self.w_d = 24.*0.0
            self.params = ["H0", "Omega_m", "Omega_k", "w0", "wa", "wb", "wc"]
            self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$w_0$", "$w_a$", "$w_b$", "$w_c$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-0.5,0.4],[-3.0,0.5],[-50.,40.],[-100.,200.],[-200.,100.]]
        elif self.model == 'XCDM' and self.nparams == 5:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = 0.0
            self.f_0 = 1.0 #parameters[2]
            self.f_a = parameters[2]
            self.f_b = parameters[3]
            self.f_c = parameters[4]
            self.params = ["H0", "Omega_m", "fa", "fb", "fc"]
            self.labels = ["$H_0$", "$\Omega_m$", "$f_a$", "$f_b$", "$f_c$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-20.,20.],[-50.,50.],[-50.,50.]]
        elif self.model == 'kXCDM' and self.nparams == 6:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_k = parameters[2]
            self.f_0 = 1.0 #parameters[2]
            self.f_a = parameters[3]
            self.f_b = parameters[4]
            self.f_c = parameters[5]
            self.params = ["H0", "Omega_m", "Omega_k", "fa", "fb", "fc"]
            self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_k$", "$f_a$", "$f_b$", "$f_c$"]
            self.priors = [[61.0,76.0],[0.1,0.5],[-0.5,0.4],[-20.,20.],[-50.,50.],[-50.,50.]]
        elif self.model == 'BW' and self.nparams == 3:
            self.H_0 = parameters[0]
            self.Omega_m = parameters[1]
            self.Omega_b = 0.0
            self.Omega_l = parameters[2]
            self.params = ["H0", "Omega_m", "Omega_l"]
            self.labels = ["$H_0$", "$\Omega_m$", "$\Omega_l$"]
            self.priors = [[50.0,90.0],[0.0,2.0],[0.0,0.5]]
        elif self.model == 'woz':
            print('Yet to define a general model ')
        else:
            print('Error: Number of parameters does not match the model specifications')
            sys.exit(0)
        
        
        if len(model)-1 == 2 and 'rd' in model and 'Mb' in model:
            self.r_d = parameters[-2]
            self.params = np.append(self.params, ["rd"]).tolist()
            self.labels = np.append(self.labels, ["$r_d$"]).tolist()
            self.priors = np.concatenate((self.priors, [[130.,160.]]),axis=0)
            self.rdsample = 'True'
            self.M_b = parameters[-1]
            self.labels = np.append(self.labels, ["$M_b$"]).tolist()
            self.params = np.append(self.params, ["Mb"]).tolist()
            self.priors = np.concatenate((self.priors, [[-20.,-19.]]),axis=0)
            self.Mbsample = 'True'
            self.Obsample = 'False'
        elif len(model)-1 == 1 and 'rd' in model:
            self.r_d = parameters[-1]
            self.params = np.append(self.params, ["rd"]).tolist()
            self.labels = np.append(self.labels, ["$r_d$"]).tolist()
            self.priors = np.concatenate((self.priors, [[100.,180.]]),axis=0)
            self.rdsample = 'True'
            self.Mbsample = 'False'
            self.Obsample = 'False'
        elif len(model)-1 == 1 and 'Mb' in model:
            self.M_b = parameters[-1]
            self.params = np.append(self.params, ["Mb"]).tolist()
            self.labels = np.append(self.labels, ["$M_b$"]).tolist()
            self.priors = np.concatenate((self.priors, [[-21.,-18.]]),axis=0)
            self.Mbsample = 'True'
            self.rdsample = 'False'
            self.Obsample = 'False'
        
        
        if len(model)-1 == 2 and 'Ob' in model and 'Mb' in model:
            self.Omega_b = parameters[-2]
            self.params = np.append(self.params, ["Omega_b"]).tolist()
            self.labels = np.append(self.labels, ["$\Omega_b$"]).tolist()
            self.priors = np.concatenate((self.priors, [[0.04,0.06]]),axis=0)
            self.M_b = parameters[-1]
            self.params = np.append(self.params, ["Mb"]).tolist()
            self.labels = np.append(self.labels, ["$M_b$"]).tolist()
            self.priors = np.concatenate((self.priors, [[-19.5,-19.]]),axis=0)
            self.Mbsample = 'True'
            self.rdsample = 'False'
            self.Obsample = 'True'
            self.params[1] = "Omega_c"
            self.labels[1] = "$\Omega_c$"
            self.priors[1] = [0.23, 0.3]
        elif len(model)-1 == 1 and 'Ob' in model:
            self.Omega_b = parameters[-1]
            self.params = np.append(self.params, ["Omega_b"]).tolist()
            self.labels = np.append(self.labels, ["$\Omega_b$"]).tolist()
            self.priors = np.concatenate((self.priors, [[0.04,0.06]]),axis=0)
            self.rdsample = 'False'
            self.Obsample = 'True'
            self.params[1] = "Omega_c"
            self.labels[1] = "$\Omega_c$"
            self.priors[1] = [0.2, 0.3]
    
        
    # All the equations here are standardly set without the radiation component
    # Define expansion rate f(z) for the corresponding dark enenrgy model
    # return np.exp(3./36.*(z*(-36.*self.w_a - 9*self.w_b*(z-2.) + self.w_c*(-6. + 3.*z -2.*z**2)) + 36.*(1. + self.w_0 + self.w_a + self.w_b/2. + self.w_c/6.)*np.log(1. +z)))
    # return np.exp(3.*(z/6.*(-6.*(self.w_a + self.w_b + self.w_c ) + 3.*z*(self.w_b + self.w_c ) -2.*self.w_c*z**2) + (1. + self.w_0 + self.w_a + self.w_b + self.w_c)*np.log(1. +z)))
    def dark_energy_f(self, z):
        # This function should define the f(z), that multiplies the \Omega_DE: which is essentially Exp(3*Int[(1+w)/(1+z)])
        if self.model in ['LCDM', 'wCDM', 'kLCDM', 'kwCDM', 'CPL', 'kCPL']:
            return np.exp(3.*(-self.w_a + self.w_a/(1. + z) + (1. + self.w_0 + self.w_a)*np.log(1. + z)))
        elif self.model in ['3CPL', 'k3CPL']:
            a = 1./(1. + z)
            intwz = -3.*((self.w_a + self.w_b/2. +self.w_c/6. + self.w_d/24.)*pow(a,3.)*pow(1.-a,1.) +
                               (3.*self.w_a + 7.*self.w_b/4. + 7.*self.w_c/12. + 7.*self.w_d/48)*pow(a,2.)*pow(1.-a,2.) +
                               (3.*self.w_a + 2.*self.w_b + 13.*self.w_c/18. + 13.*self.w_d/72.)*pow(a,1.)*pow(1.-a,3.) +
                               (self.w_a + 3.*self.w_b/4. + 11.*self.w_c/36. + 25.*self.w_d/288.)*pow(1.-a,4.) +
                               (1. + self.w_0 + self.w_a + self.w_b/2. + self.w_c/6. + self.w_d/24.)*np.log(a))
            if intwz > 10.0:
                return np.exp(10)
            else:
                return np.exp(intwz)
        elif self.model in ['kXCDM','XCDM']:
            a = 1./(1. + z)
            return self.f_0 + self.f_a*pow((1.-a),1.) + self.f_b*pow((1.-a),2.) + self.f_c*pow((1.-a),3.)


    def dark_energy_w(self, z):
        # This function should define the f(z), that multiplies the \Omega_DE: which is essentially Exp(3*Int[(1+w)/(1+z)])
        if self.model in ['LCDM', 'wCDM', 'kLCDM', 'kwCDM', 'CPL', 'kCPL']:
            a = 1./(1. + z)
            return self.w_0 + self.w_a*(1-a)
        elif self.model in ['3CPL', 'k3CPL']:
            a = 1./(1. + z)
            # return self.w_0 + self.w_a*(1.-a) + self.w_b*pow((1.-a),2.) + self.w_c*pow((1.-a),3.) + self.w_d * pow((1.-a),4.)
            return self.w_0 + self.w_a*(1.-a) + self.w_b*pow((1.-a),2.)/2. + self.w_c*pow((1.-a),3.)/6. + self.w_d * pow((1.-a),4.)/24.
        elif self.model in ['kXCDM','XCDM']:
            a = 1./(1. + z)
            return -1. + ( self.f_a * pow(1. + z,2) + z * (3.* self.f_c * z + 2. * self.f_b * (1. + z)) ) / ( 3. * (self.f_0 * pow(1. + z,3) + z * ( self.f_a * pow(1. + z,2) + z * ( self.f_b * (1. + z) + z * self.f_c ) ) ))

    # Define expansion rate H(z) for the corresponding models
    def hubble_rate(self, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        zeq = 2.5 * 10.**4. * (self.Omega_m + self.Omega_b) * (self.H_0/100.)**2. /(2.7255/2.7)**4.
        self.z_eq = zeq
        #Omega_r = (self.Omega_m + self.Omega_b) /(1. + zeq)
        Omega_r = 4.18343*10**-5./(self.H_0/100.)**2.
        if self.model in ['LCDM', 'wCDM', 'kLCDM', 'kwCDM', 'CPL', '3CPL', 'kCPL', 'k3CPL', 'XCDM', 'kXCDM']:
            E2 = Omega_r*pow((1. + z),4.) + self.Omega_m*pow((1. + z),3.) + self.Omega_b*pow((1. + z),3.) + self.Omega_k*pow((1. + z),2.) + (1. - self.Omega_m - self.Omega_k - self.Omega_b - Omega_r)*self.dark_energy_f(z)
            Hofz = self.H_0*np.sqrt( E2 )
            return np.nan_to_num(Hofz)
        elif self.model in ['BW']:
            Omega_L = 1. - self.Omega_m + 2*np.sqrt(self.Omega_l)
            Hofz = self.H_0*np.sqrt( self.Omega_m*pow((1. + z),3.) + Omega_L + 2.*self.Omega_l -  2*np.sqrt(self.Omega_l*(self.Omega_m*pow((1. + z),3.) + Omega_L + self.Omega_l)))
            return np.nan_to_num(Hofz)
        elif self.model in ['bCPL', 'woz']:
            print('This equation has still to be defined ')
            return 1.
   
    
    # Define transverse comoving distance D_M(z) accordingly for various models
    def transverse_distance(self, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        def Ly(y,t):
            # Here this could be replaced with expansion_rate_int if needed 
            return 1./self.hubble_rate(t)

        y=it.odeint(Ly,0.0,z)
        tint = np.array(y[:,0])
            
            
        if self.model in ['LCDM', 'wCDM', 'CPL', '3CPL', 'XCDM', 'BW']:
            return np.nan_to_num( self.clight* tint)
        elif self.model in ['kLCDM', 'kwCDM', 'kCPL', 'k3CPL', 'kXCDM']:
            if self.Omega_k > 0.0:
                return np.nan_to_num( self.clight/(np.sqrt(self.Omega_k)*self.H_0)*np.sinh(np.sqrt(self.Omega_k) * self.H_0*tint))

            elif self.Omega_k < 0.0:
                return np.nan_to_num( self.clight/(np.sqrt(abs(self.Omega_k))*self.H_0)*np.sin((np.sqrt(abs(self.Omega_k)))* self.H_0* tint))
                
            elif self.Omega_k == 0.0:
                return np.nan_to_num( self.clight* tint)
        elif self.model in ['bCPL', 'woz']:
            print('This integral has still to be defined')
            return 1.

    #To define the value of r_d using the Aubourg15 formulae
    def sound_horizon(self):
        if self.rdsample == 'True':
            rd = self.r_d
        else:
            m_nu = 0.06 # In the units of eV
            omega_nu = 0.0107 *(m_nu/1.0) #This is in the units of eV. This should be equal to 6.42*10^(-4)
            if self.Obsample == 'True':
                omega_b = self.Omega_b*(self.H_0/100.)**2. #0.0217
            else:
                omega_b = 0.0217
            
            omega_cb = (self.Omega_m+self.Omega_b)*(self.H_0/100.)**2 - omega_nu
            if omega_cb < 0:
                rd = -1.0
            else:
                rd = np.nan_to_num(55.154 * np.exp(-72.3* pow(omega_nu + 0.0006,2))/(pow(omega_cb, 0.25351)*pow(omega_b, 0.12807)))

        return rd

    # This is to model the mu, but however the Mb is set inside, therefore it is M_b + mu but not mu alone
    def distance_modulus(self, z1, z2):
        if self.Mbsample == 'True':
            Mb = self.M_b
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(z1) )
        else:
            # This is useful if only SN data is needed to be used
            Mb = -19.05
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(z1) )
        return mu
    
    # This is to provide the Mb alone
    def Abs_M(self):
        if self.Mbsample == 'True':
            Mb = self.M_b
        else:
            # This is useful if only SN data is needed to be used
            Mb = -19.05
        return Mb

    def z_rec(self):
        '''
        redshift of recombination, from Hu & Sugiyama 1996 fitting formula
        '''
        
        omega_b = self.Omega_b*(self.H_0/100.)**2.
        omega_m = self.Omega_m*(self.H_0/100.)**2. + omega_b
        g1 = 0.0738*omega_b**(-0.238)/(1.+39.5*omega_b**0.763)
        g2 = 0.560/(1.+21.1*omega_b**1.81)
        return 1048.*(1.+0.00124*omega_b**(-0.738))*(1.+g1*omega_m**g2)

    def rs(self,z):
        '''
        sound horizon at z, the variable is x = ln(a)
        '''
        omega_b = self.Omega_b*(self.H_0/100.)**2.
        Rs = 31500./(2.7255/2.7)**4. #3./4./2.47282 *1e+05   # just photons are coupled to baryons
        return self.clight/self.H_0 * quad(func=lambda x: np.exp(-x)/(self.expansion_rate(np.exp(-x)-1.)*np.sqrt(3.*(1. + Rs * omega_b * np.exp(x) )) ), a = np.log(1e-40) , b = np.log(1./(1.+z)) , limit=100 )[0]


    def rs2(self,z):
        '''
        sound horizon at z, the variable is x = a
        '''
        omega_b = self.Omega_b*(self.H_0/100.)**2.
        Rs = 31500./(2.7255/2.7)**4. #3./4./2.47282 *1e+05   # just photons are coupled to baryons
        return self.clight/self.H_0 * quad(func=lambda x: 1./(x**2. * self.expansion_rate(1./x - 1.)*np.sqrt(3.*(1. + Rs * omega_b * x )) ), a = 0 , b = 1./(1. + z))[0]

    # Use the time whenever needed to assert the speed of the code.
    # time0 = time.time()
    # time1=time.time()
    # print time1-time0
