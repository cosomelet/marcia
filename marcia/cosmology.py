import numpy as np
import scipy.integrate as it
from scipy.integrate import quad
from marcia.params import Params


def Cosmology(model,parameters,prior_file=None):
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
        """Part of sub class"""
        pass

    def dark_energy_w(self,parameters, z):
        """Part of sub class"""
        pass

    def transverse_distance(self,parameters,z):
        """Part of sub class"""
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
    

class wCDM(Cosmology_base):
    def __init__(self, params,prior_file=None):
        super().__init__(params,prior_file)
        self.__check_mandatory_parameters__()

    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'wCDM: H0 is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'wCDM: w0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'wCDM: Omega_m is not defined in the parameters'
        assert len(self.param.parameters) == 3, 'wCDM: parameters are not correct'
        
    
    def dark_energy_f(self, parameters, z):
        p = self.param(parameters)
        return np.exp(3.*(-p.wa + p.wa/(1. + z) + (1. + p.w0 + p.wa)*np.log(1. + z)))
    
    def dark_energy_w(self,parameters, z):
        p = self.param(parameters)
        a = self.a(z)
        return p.w0 + p.wa*(1-a)
    
    def transverse_distance(self,parameters, z):
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
        assert len(self.param.parameters) == 2, 'LCDM: parameters are not correct'

class CPL(wCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'CPL: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'CPL: Omega_m is not defined in the parameters'
        assert 'wa' in self.param.parameters, 'CPL: wa is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'CPL: w0 is not defined in the parameters'
        assert len(self.param.parameters) == 4, 'CPL: parameters are not correct'

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
        assert len(self.param.parameters) == 6, 'CPL3: parameters are not correct'

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
        assert len(self.param.parameters) == 5, 'XCDM: parameters are not correct'
    
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
        assert len(self.param.parameters) == 4, 'kwCDM: parameters are not correct'

    
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
            
        elif self.Omega_k == 0.0:
            return np.nan_to_num( self.clight* tint)
    

class kLCDM(kwCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kLCDM: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kLCDM: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kLCDM: Omega_k is not defined in the parameters'
        assert len(self.param.parameters) == 3, 'kLCDM: parameters are not correct'

class kCPL(kwCDM):
    def __init__(self, parameters,prior_file=None):
        super().__init__(parameters,prior_file)
    
    def __check_mandatory_parameters__(self):
        assert 'H0' in self.param.parameters, 'kCPL: H0 is not defined in the parameters'
        assert 'Omega_m' in self.param.parameters, 'kCPL: Omega_m is not defined in the parameters'
        assert 'Omega_k' in self.param.parameters, 'kCPL: Omega_k is not defined in the parameters'
        assert 'w0' in self.param.parameters, 'kCPL: w0 is not defined in the parameters'
        assert 'wa' in self.param.parameters, 'kCPL: wa is not defined in the parameters'
        assert len(self.param.parameters) == 5, 'kCPL: parameters are not correct'
    
        
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
        assert len(self.param.parameters) == 7, 'kCPL3: parameters are not correct'
    
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
            
        elif self.Omega_k == 0.0:
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
        assert len(self.param.parameters) == 6, 'kXCDM: parameters are not correct'
    
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
            
        elif self.Omega_k == 0.0:
            return np.nan_to_num( self.clight* tint)
        
