import numpy as np
import scipy.integrate
from scipy.integrate import quad
import scipy.constants as const
import sys
import time
import cmath
import os
from marcia.database import Data

# In this file we compute comsologycal quantities such as expansion rate and sound horizon etc.. 

class Kernels(object):
    """
        General background theory that all the theories must be inherited from.
        Inputs would be the appropriate cosmological parameters needed for the particular model. 
        
        """
    
    def __init__(self, model, parameters, data):
        
        self.data = data 
        self.model = model
        self.params = parameters
        
        if len(self.data) != len(self.model):
            print('Error: data length does not match the number of kernels')
            
        
        if 'self-scaled' in model and len(model) == len(parameters)+1:
            self.model = self.model[:-1]
            # Implying that the scale length for all the datasets is set to be the same, we need sigma_f equal to the number of GPs and one l_f
            for i in range(len(parameters) - len(model)): 
                self.params = np.append(self.params, parameters[-1])
            self.params = np.transpose(np.reshape(np.params, (len())))
        elif self.params != 2*len(models):
            # We need atleast two paramters per GP: {sigma_f, l_f}
            print('Error: Number of parameters does not match the model specifications')
        
        # Everytime this is initialised I set the sigma_f_check and l_check to avoid computing the
        # covariance matrices for every single call with same hyperparameters
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

    