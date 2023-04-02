import numpy as np
import scipy.integrate
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
        
        """
    
    def __init__(self, model, parameters,data_cc=None):
        if data_cc is None:
            self.data_cc = Data('CC')
        else:
             #TODO: implement data_cc
            raise NotImplementedError('Data_cc is not implemented yet')
        # self.zlist = z
        # To add the expected nuisance parameters of the model here
        self.model = model[0]
        if len(model) > 1:
            self.nparams = len(parameters) - (len(model) -1)
        else:
            self.rdsample = 'False'
            self.Mbsample = 'False'
            self.nparams = len(parameters)
        
        # Everytime this is initialised I set the sigma_f_check and l_check to avoid computing the
        #   covariance matrices for every single call with same hyperparameters
        self.sigma_f_check = -1.0
        self.l_check = -1.0
        
        self.clight = 299792458. / 1000.
        # Maybe here I would like to add a first check for verifying number of parameters.
        if self.model in ['SE', 'M92', 'M72'] and self.nparams == 2:
            self.Omega_k = 0.0
            self.s_f = pow(10.,parameters[0])
            self.l = pow(10.,parameters[1])
            self.labels = ["$\log_{10}(\sigma_f)$", "$\log_{10}(l)$"]
            self.priors = [[-1.0,3.5],[-4.0,2.0]]
        elif self.model == 'kernel':
            print('Yet to define a general model ')
        else:
            print('Error: Number of parameters does not match the model specifications')
            sys.exit(0)
        
        
        if len(model) == 4:
            self.Omega_k = parameters[-3]
            self.labels = np.append(self.labels, ["$\Omega_k$"])
            self.priors = np.concatenate((self.priors, [[-0.5,0.5]]),axis=0)
            self.r_d = parameters[-2]
            self.labels = np.append(self.labels, ["$r_d$"])
            self.priors = np.concatenate((self.priors, [[100.,180.]]),axis=0)
            self.rdsample = 'True'
            self.M_b = parameters[-1]
            self.labels = np.append(self.labels, ["$M_b$"])
            self.priors = np.concatenate((self.priors, [[-21.,-18.]]),axis=0)
            self.Mbsample = 'True'
    
        self.DetCovMat = 0.0
        self.InvCovMat = self.Inv_Cov_Mat()
        self.InvCovMatCC = np.dot(self.InvCovMat, self.data_cc.cc).T
        self.Omega_m = 0.3 # @sandeep review this
        self.H_0 = 70.0 # @sandeep review this

    # This is to define the different possible kernel choices
    def kernel(self, x1, x2):
        if self.model == 'SE':
            return self.s_f**2. * np.exp(- ((x1-x2)**2.)/(2. * self.l**2.) )
        elif self.model == 'M92':
            return self.s_f**2. * np.exp( -np.sqrt( ((x1-x2)**2.)/(2. * self.l**2.) ) )
        elif self.model == 'M72':
            return self.s_f**2. * np.exp( -np.sqrt( ((x1-x2)**2.)/(2. * self.l**2.) ) )

    # To define the all the basis functions of all the possible kernels
    def basis_function(self, x1,x2):
        if self.model == 'SE':
            return self.s_f**2. * np.exp(- ((x1-x2)**2.)/(2. * self.l**2.) )
        elif self.model == 'M92':
            return self.s_f**2. * np.exp( -np.sqrt( ((x1-x2)**2.)/(2. * self.l**2.) ) )
        elif self.model == 'M72':
            return self.s_f**2. * np.exp( -np.sqrt( ((x1-x2)**2.)/(2. * self.l**2.) ) )

    #Covarinace matrix of data
    def Inv_Cov_Mat(self):
        z_CC = self.data_cc.z
        Sigma_CC = np.diag(self.data_cc.dcc**2)
        cmatrix = np.reshape([0.0]*(len(z_CC)*len(z_CC)),(len(z_CC),len(z_CC)))
        for i in range(len(z_CC)):
            for j in range(i+1):
                #print self.kernel(z_CC[i],z_CC[j])
                cmatrix[i,j] = self.kernel(z_CC[i],z_CC[j])
        out = cmatrix.T + cmatrix
        np.fill_diagonal(out,np.diag(cmatrix))
        covmat = np.array(out) + np.array(Sigma_CC)
        self.DetCovMat = np.linalg.det(covmat)
        return np.linalg.inv(covmat)

    # Covariance matrix of prediction star
    def Cov_Mat_S(self, x1):
        z_CC = self.data_cc.z
        cmat_Star = [0.0]*(len(z_CC))
        for i in range(len(z_CC)):
            cmat_Star[i] = self.kernel(x1,z_CC[i])
        return cmat_Star
    
    # Covariance matrix of prediction star
    def Cov_Mat_SS(self, x1):
        zp = [x1]
        #np.linspace(0.,2.5,50)
        cmatrix = np.reshape([0.0]*(len(zp)*len(zp)),(len(zp),len(zp)))
        for i in range(len(zp)):
            for j in range(i+1):
                cmatrix[i,j] = self.kernel(zp[i],zp[j])
        cmat_Star_Star = cmatrix.T + cmatrix
        np.fill_diagonal(cmat_Star_Star,np.diag(cmatrix))

        return cmat_Star_Star

    # Define expansion rate H(z) for the corresponding models
    def hubble_rate_var(self, z:list):
        
        # Use this print, in case needed for Error_handling.
        if self.model in ['SE', 'M92', 'M72']:
            #return self.H_0*np.sqrt( self.Omega_m*(1. + self.z)**3 + self.Omega_k*(1. + self.z)**2 + (1. - self.Omega_m - self.Omega_k)*self.dark_energy_f() )
            meanf = np.dot(np.array(self.Cov_Mat_S(z)).T, self.InvCovMatCC )
            covf = self.Cov_Mat_SS(z) - np.dot(np.dot(np.array(self.Cov_Mat_S(z)).T, self.InvCovMat),np.array(self.Cov_Mat_S(z)))
            #Hofz = np.dot(np.array(self.Cov_Mat_S(z)), self.InvCovMatCC )
            Hofz = np.reshape(np.random.multivariate_normal(meanf, covf,1),(len(z),))
            return np.nan_to_num(Hofz)
        elif self.model in ['kernel']:
            print('This equation has still to be defined ')
            return 1.

    # Define expansion rate E(z) for the corresponding models. This is simply H(z)/H_0 but I prefer to define seperately
    def expansion_rate(self, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        if self.model in ['SE', 'M92', 'M72']:
            Eofz = self.hubble_rate_var(z)/self.hubble_rate_var(0.0)
            return np.nan_to_num(Eofz)
        elif self.model in ['kernel']:
            print('This equation has still to be defined ')
            return 1.
    
    # Define transverse comoving distance D_M(z) accordingly for various models
    def transverse_distance(self, z):
        # Use this print, in case needed for Error_handling.
        # print ''
        if self.model in ['SE', 'M92', 'M72']:
            if self.Omega_k > 0.0:
                return np.nan_to_num( self.clight/(np.sqrt(self.Omega_k)*self.hubble_rate_var(0.0))*np.sinh(np.sqrt(self.Omega_k)*self.hubble_rate_var(0.0)*quad(lambda y: 1./self.hubble_rate_var(y),0,z)[0]))
            elif self.Omega_k < 0.0:
                return np.nan_to_num( self.clight/(np.sqrt(abs(self.Omega_k))*self.hubble_rate_var(0.0))*np.sin(np.sqrt(abs(self.Omega_k))*self.hubble_rate_var(0.0)*quad(lambda y: 1./self.hubble_rate_var(y),0,z)[0]))
            elif self.Omega_k == 0.0:
                return np.nan_to_num( self.clight*quad(lambda y: 1./self.hubble_rate_var(y),0,z)[0])
        elif self.model in ['kernel']:
            print('This integral has still to be defined')
            return 1.

    #To define the value of r_d using the Aubourg15 formulae
    def sound_horizon(self):
        if self.rdsample == 'True':
            rd = self.r_d
        else:
            m_nu = 0.06 # In the units of eV
            omega_nu = 0.0107 *(m_nu/1.0) #This is in the units of eV. This should be equal to 6.42*10^(-4)
            omega_b = 0.0217
            omega_cb = self.Omega_m*(self.H_0/100.)**2 - omega_nu
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
            # This is useful if only SN data is needed tp be used
            Mb = -19.05
            mu = Mb + 25. + 5.*np.log10( (1+ z2)* self.transverse_distance(z1) )
        return mu



