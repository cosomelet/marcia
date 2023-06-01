import os 
import numpy as np
import toml
import scipy.integrate as it
from scipy.integrate import quad
from marcia.params import Params
from scipy.interpolate import interp1d
from marcia.backend import cosmology  as cbackend




def Cosmology(model:str,parameters:list,prior_file:str=None):
    """
    Calculate cosmological parameters for a given cosmology

    Parameters
    ----------
    model : string
        Name of cosmological model to use.
    parameters : list
        List of model parameters to calculate. Can include
        'H0': Hubble constant
        'Omega_m': matter density parameter
        'Omega_b': baryon density parameter
    prior_file : string
        Name of file containing priors on parameters

    Returns
    -------
    cosmological model object
    """
    models = ['LCDM','wCDM','CPL','CPL3','kwCPL','kCPL3','kLCDM','XCDM','kXCDM']
    if model not in models:
        raise ValueError(
            f"""Requested model not available
            Available models are: {models}
            """
            )
    return globals()[model](parameters,prior_file)

class Cosmology_base(object):
    """
    General background theory that all the theories must be inherited from.
    Inputs would be the appropriate cosmological parameters needed for the particular model. 
    To change
    """
    
    def __init__(self, params,prior_file=None):

        self.param = Params(params,prior_file)
        self.priors = self.param.Priors 
        self.labels = self.param.Labels


        self.rdsample = False
        self.Mbsample = False
        self.Obsample = False
        
        self.const = {}
        const = {}
        const = toml.load(os.path.join(os.path.dirname(__file__), 'constants.ini'))['constants']
        for key, value in const.items():
            self.const[key] = value

        if 'r_d' in params:
            self.rdsample = True
        if 'M_b' in params:
            self.Mbsample = True
        if 'Omega_b' in params:
            self.Obsample = True


    
        if self.rdsample and self.Obsample:
            raise ValueError('rd and Ob cannot be sampled together')
        

        self.clight = 299792458. / 1000.
    
    @staticmethod
    def a(z):
        return 1. / (1. + z)

    def dark_energy_f(self,parameters, z):
        """Part of sub class"""
        pass

    def dark_energy_w(self,parameters, z):
        """Part of sub class"""
        pass

    def _transverse_distance_(self,parameters,z):
        """Part of sub class"""
        pass
    
    def transverse_distance(self,parameters,z):
        z_inp = cbackend.z_inp(z)
        d = self._transverse_distance_(parameters,z_inp)
        return cbackend.interpolate(z_inp,z,d)

    def hubble_rate(self,parameters, z):
        p = self.param(parameters)
        de = self.dark_energy_f(parameters,z)
        return cbackend.hubble_rate(p.H0,p.Omega_m,p.Omega_b,p.Omega_k,de, z)

    #To define the value of r_d using the Aubourg15 formula

    def sound_horizon(self,parameters):
        p = self.param(parameters)
        if self.rdsample:
            return p.r_d
        else:
            return cbackend.sound_horizon(p.H0,p.Omega_m,p.Omega_b,self.Obsample)

    # This is to model the mu, but however the Mb is set inside, therefore it is M_b + mu but not mu alone
    def distance_modulus(self,parameters,z1, z2):
        p = self.param(parameters)
        if self.Mbsample:
            Mb = p.M_b
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(parameters, z1) )
        else:
            # This is useful if only SN data is needed to be used
            Mb = -19.05
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(parameters, z1) )
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
        return self.clight/p.H0 * quad(func=lambda x: np.exp(-x)/(self.hubble_rate(parameters, np.exp(-x)-1.)/p.H0 * np.sqrt(3.*(1. + Rs * omega_b * np.exp(x) )) ), a = np.log(1e-40) , b = np.log(1./(1.+z)) , limit=100 )[0]


    def rs2(self,parameters,z):
        '''
        sound horizon at z, the variable is x = a
        '''
        p = self.param(parameters)
        omega_b = p.Omega_b*(p.H0/100.)**2.
        Rs = 31500./(2.7255/2.7)**4. #3./4./2.47282 *1e+05   # just photons are coupled to baryons
        return self.clight/p.H0 * quad(func=lambda x: 1./(x**2. * self.hubble_rate(parameters, 1./x - 1.)/p.H0 * np.sqrt(3.*(1. + Rs * omega_b * x )) ), a = 0 , b = 1./(1. + z))[0]
    

class wCDM(Cosmology_base):
    def __init__(self, params,prior_file=None):
        super().__init__(params,prior_file)
        self.__check_mandatory_parameters__()

    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'wCDM: H0 is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'wCDM: w0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'wCDM: Omega_m is not defined in the parameters'
        n=3
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'wCDM: parameters are not correct'
    
    def dark_energy_f(self, parameters, z):
        p = self.param(parameters)
        return cbackend.dark_energy_f_wCDM(p.w0, p.wa, z)
    
    def dark_energy_w(self,parameters, z):
        p = self.param(parameters)
        a = self.a(z)
        return p.w0 + p.wa*(1-a)
    
    def _transverse_distance_(self,parameters, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        def Ly(y,t):
            # Here this could be replaced with expansion_rate_int if needed 
            return 1./self.hubble_rate(parameters,t)

        y=it.odeint(Ly,0.0,z)
        tint = np.array(y[:,0])
        return np.nan_to_num( self.clight* tint)
    
class LCDM(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'LCDM: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'LCDM: Omega_m is not defined in the parameters'
        n=2
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'LCDM: parameters are not correct'

class CPL(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'CPL: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'CPL: Omega_m is not defined in the parameters'
        assert 'wa' in self.param.parameters, 'CPL: wa is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'CPL: w0 is not defined in the parameters'
        n=4
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'CPL: parameters are not correct'

class CPL3(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'CPL3: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'CPL3: Omega_m is not defined in the parameters'
        assert 'wa' in self.param.parameters, 'CPL3: wa is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'CPL3: w0 is not defined in the parameters'
        assert 'wb' in self.param.parameters, 'CPL3: wb is not defined in the parameters'
        assert 'wc' in self.param.parameters, 'CPL3: wc is not defined in the parameters'
        n=6
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'CPL3: parameters are not correct'

    def dark_energy_f(self,parameters, z):
        p = self.param(parameters)
        a = self.a(z)
        intwz = -3.*((p.wa + p.wb/2. +p.wc/6. + p.wd/24.)*pow(a,3.)*pow(1.-a,1.) +
                            (3.*p.wa + 7.*p.wb/4. + 7.*p.wc/12. + 7.*p.wd/48)*pow(a,2.)*pow(1.-a,2.) +
                            (3.*p.wa + 2.*p.wb + 13.*p.wc/18. + 13.*p.wd/72.)*pow(a,1.)*pow(1.-a,3.) +
                            (p.wa + 3.*p.wb/4. + 11.*p.wc/36. + 25.*p.wd/288.)*pow(1.-a,4.) +
                            (1. + p.w0 + p.wa + p.wb/2. + p.wc/6. + p.wd/24.)*np.log(a))
        if intwz > 10.0:
            return np.exp(10)
        else:
            return np.exp(intwz)
        
    def dark_energy_w(self, parameters,z):
        p = self.param(parameters)
        a = self.a(z)
        return p.w0 + p.wa*(1.-a) + p.wb*pow((1.-a),2.)/2. + p.wc*pow((1.-a),3.)/6. + p.wd * pow((1.-a),4.)/24.

class XCDM(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'XCDM: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'XCDM: Omega_m is not defined in the parameters'
        assert 'fa' in self.param.parameters, 'XCDM: fa is not defined in the parameters'
        assert 'fb' in self.param.parameters, 'XCDM: fb is not defined in the parameters'
        assert 'fc' in self.param.parameters, 'XCDM: fc is not defined in the parameters'
        n=5
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'XCDM: parameters are not correct'
    
    def dark_energy_f(self,parameters,z):
        p = self.param(parameters)
        a = self.a(z)
        return p.f0 + p.fa*pow((1.-a),1.) + p.fb*pow((1.-a),2.) + p.fc*pow((1.-a),3.)


    def dark_energy_w(self,parameters, z):
        p = self.param(parameters)
        a = self.a(z)
        return -1. + ( p.fa * pow(1. + z,2) + z * (3.* p.fc * z + 2. * p.fb * (1. + z)) ) / ( 3. * (p.f0 * pow(1. + z,3) + z * ( p.fa * pow(1. + z,2) + z * ( p.fb * (1. + z) + z * p.fc ) ) ))

class kwCDM(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)

    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kwCDM: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kwCDM: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kwCDM: Omega_k is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'kwCDM: w0 is not defined in the parameters'
        n=4
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'kwCDM: parameters are not correct'

    
    def transverse_distance(self, parameters, z):
        p = self.param(parameters)
        def Ly(y,t):
            return 1./self.hubble_rate(parameters,t)

        y=it.odeint(Ly,0.0,z)
        tint = np.array(y[:,0])
        if p.Omega_k > 0.0:
            return np.nan_to_num( self.clight/(np.sqrt(p.Omega_k)*p.H0)*np.sinh(np.sqrt(p.Omega_k) * p.H0*tint))

        elif p.Omega_k < 0.0:
            return np.nan_to_num( self.clight/(np.sqrt(abs(p.Omega_k))*p.H0)*np.sin((np.sqrt(abs(p.Omega_k)))* p.H0* tint))
            
        elif p.Omega_k == 0.0:
            return np.nan_to_num( self.clight* tint)
    

class kLCDM(kwCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kLCDM: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kLCDM: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kLCDM: Omega_k is not defined in the parameters'
        n=3
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'kLCDM: parameters are not correct'

class kCPL(kwCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kCPL: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kCPL: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kCPL: Omega_k is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'kCPL: w0 is not defined in the parameters'
        assert 'wa' in self.param.parameters, 'kCPL: wa is not defined in the parameters'
        n=5
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'kCPL: parameters are not correct'
    
class kCPL3(CPL3):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kCPL3: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kCPL3: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kCPL3: Omega_k is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'kCPL3: w0 is not defined in the parameters'
        assert 'wa' in self.param.parameters, 'kCPL3: wa is not defined in the parameters'
        assert 'wc' in self.param.parameters, 'kCPL3: wc is not defined in the parameters'
        assert 'wb' in self.param.parameters, 'kCPL3: wb is not defined in the parameters'
        n=7
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'kCPL3: parameters are not correct'
    
    def transverse_distance(self, parameters, z):
        p = self.param(parameters)
        def Ly(y,t):
            return 1./self.hubble_rate(parameters,t)

        y=it.odeint(Ly,0.0,z)
        tint = np.array(y[:,0])
        if p.Omega_k > 0.0:
            return np.nan_to_num( self.clight/(np.sqrt(p.Omega_k)*p.H0)*np.sinh(np.sqrt(p.Omega_k) * p.H0*tint))

        elif p.Omega_k < 0.0:
            return np.nan_to_num( self.clight/(np.sqrt(abs(p.Omega_k))*p.H0)*np.sin((np.sqrt(abs(p.Omega_k)))* p.H0* tint))
            
        elif p.Omega_k == 0.0:
            return np.nan_to_num( self.clight* tint)

    
class kXCDM(XCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kXCDM: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kXCDM: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kXCDM: Omega_k is not defined in the parameters'
        assert 'fa' in self.param.parameters, 'kXCDM: fa is not defined in the parameters'
        assert 'fb' in self.param.parameters, 'kXCDM: fb is not defined in the parameters'
        assert 'fc' in self.param.parameters, 'kXCDM: fc is not defined in the parameters'
        n=6
        if self.rdsample or self.Obsample:
            n+=1
        if self.Mbsample:
            n+=1
        assert len(self.param.parameters) == n, 'kXCDM: parameters are not correct'
    
    def transverse_distance(self, parameters, z):
        """transverse distance in Mpc/h"""
        p = self.param(parameters)
        def Ly(y,t):
            return 1./self.hubble_rate(parameters,t)

        y=it.odeint(Ly,0.0,z)
        tint = np.array(y[:,0])
        if p.Omega_k > 0.0:
            return np.nan_to_num( self.clight/(np.sqrt(p.Omega_k)*p.H0)*np.sinh(np.sqrt(p.Omega_k) * p.H0*tint))

        elif p.Omega_k < 0.0:
            return np.nan_to_num( self.clight/(np.sqrt(abs(p.Omega_k))*p.H0)*np.sin((np.sqrt(abs(p.Omega_k)))* p.H0* tint))
            
        elif p.Omega_k == 0.0:
            return np.nan_to_num( self.clight* tint)
        
