import sys
import cmath
import numpy as np
import scipy as sp
from scipy.integrate import quad
import scipy.constants as const
import time

from marcia import Data


class Kernels(object):
    """
        General Gaussian process modelling that all the specific kernels must be inherited from.
        Inputs would be the appropriate amplitude and scaling parameters needed for the particular model. 

        """

    def __init__(self, model, parameters, data):

        self.clight = 299792458. / 1000.  # km/s

        self.kernels = ['SE', 'M92', 'M72']  # Possible kernel choices

        # The data must contain the list/lists of x-axes
        self.data = data
        self.ndata = len(data)
        # The possible model cases are ['SE', 'SE', 'SE'] etc. for 3 tasks and ['SE', 'SE', 'SE', 'self-scaled'] for 3 tasks and a self-scaled GP
        self.model = model
        self.nmodel = len(model)
        self.params = parameters
        self.nparams = len(parameters)

        if 'self-scaled' in self.model and (self.nparams == self.nmodel):
            self.ntasks = self.nmodel - 1
            self.model = self.model[:-1]
            # Implying that the scale length for all the datasets is set to be the same, we need sigma_f equal to the number of GPs and one l_s
            for i in range(self.ntasks-1):
                self.params = np.append(self.params, parameters[-1])

        elif 'self-scaled' not in model and (self.nparams == 2*len(self.model)):
            # We have the number of parameters matching the number of GP tasks
            self.ntasks = len(self.model)

        elif 'self-scaled' not in model and (self.nparams != 2*len(model)):
            # We need atleast two paramters per GP task: {sigma_f, l_s}
            print('Error: Number of parameters does not match the model specifications')
            sys.exit(0)

        # Reshaping the parameters to be a 2D array of shape (2, ntasks)
        self.params = np.transpose(np.reshape(self.params, (2, self.ntasks)))

        # To create a dictionary of paramters and datasets
        self.data_f = {}
        self.sig_f = {}
        self.l_s = {}
        self.kernel_f = {}
        self.CovMat = {}
        self.CovMat_all = []

        if len(self.data) != self.ntasks:
            print('Error: data length does not match the number of kernels')
            sys.exit(0)
        else:
            for i in range(self.ntasks):
                if self.model[i] in self.kernels and len(self.params[i]) == 2:
                    self.kernel_f[f'task_{i}'] = self.model[i]
                    self.sig_f[f'task_{i}'] = self.params[i, 0]
                    self.l_s[f'task_{i}'] = self.params[i, 1]
                    # To create a dictionary of data
                    self.data_f[f'task_{i}'] = self.data[i]

                else:
                    print(self.model[i], self.params[i])
                    print('Error: model or parameters not specified correctly')
                    sys.exit(0)

        # self.CovMat = self.Cov_Mat()


    def matern(self, nu, x1, x2, l_s):
        """ 
            Defines the generalised matern kernel https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
            nu is a positive half-integer that determines the differentiability of the function and the smoothness of the resulting Gaussian process sample paths.
            nu = 1/2, 3/2, 5/2, ... tend to be used most often in practice.
            nu tends to infinity, the matern kernel tends to the squared exponential kernel
            nu tends to 0, the matern kernel tends to the absolute exponential kernel
            The input x1 and x2 are the x-axes of the data points and l_s is the scale length
            """
        
        # This fucntion returns nan when x1 = x2, so we need to replace nan with zero 
        return np.nan_to_num((2.**(1.-nu)/sp.special.gamma(nu)) * ((np.sqrt(2.*nu) * np.abs(x1-x2))/l_s)**nu * sp.special.kv(nu, (np.sqrt(2.*nu) * np.abs(x1-x2))/l_s)) + np.eye(len(x1)) # np.nan_to_num() to replace nan with zero    

    
    def kernel(self, model, params, x1, x2=None):
        """ 
            Defines the kernel function for the GP model and returns the covariance matrix 
            """
        if x2 is None:
            x2 = x1
        if model == 'SE': # Squared Exponential kernel
            return params[0]**2. * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(2. * params[1]**2.))
        if model == 'M92': # Matern 9/2 kernel
            mat_nu = 9./2.
            return params[0]**2. * self.matern(mat_nu, x1[:, None], x2[None, :], params[1])
        if model == 'M72': # Matern 7/2 kernel
            mat_nu = 7./2.
            return params[0]**2. * self.matern(mat_nu, x1[:, None], x2[None, :], params[1]) 


    def cross_kernel(self, model1, model2, params1, params2, x1, x2):
        """ 
            Defines the cross kernel function for the GP model and returns the covariance matrix 
            """
        # product of the sigmas
        sig1sig2 = params1[0] * params2[0]
        # product of the scale lengths
        l1l2 = params1[1] * params2[1]
        # quadrature of length scales
        l1l2_quad = params1[1]**2. + params2[1]**2.
        # difference between x1 and x2
        x1x2 = x1[:, None]-x2[None, :]

        if model1 == 'SE' and model2 == 'SE': # Squared Exponential kernel
            # This fucntion boils down to the SE kernel function when the scale lengths are equal for both the datasets
            return sig1sig2 * np.exp(- (x1x2**2.)/(l1l2_quad)) * np.sqrt(2. * l1l2 / l1l2_quad)
        else:
            print('Error: Cross kernel not defined')
            sys.exit(0)

    def Cov_Mat(self):
        """ 
            Defines the covariance matrix for the GP model and returns the covariance matrix 
            """
        # To create a dictionary of covariance matrices for each task
        
        for i in range(self.ntasks):
            for j in range(self.ntasks):
                if i == j:
                    self.CovMat['task_{0}{1}'.format(i, j)] = self.kernel(self.kernel_f['task_{0}'.format(i)], self.params[i], self.data_f['task_{0}'.format(i)])
                elif i < j:
                    self.CovMat['task_{0}{1}'.format(i, j)] = np.transpose(self.cross_kernel(self.kernel_f['task_{0}'.format(i)], self.kernel_f['task_{0}'.format(j)], self.params[i], self.params[j], self.data_f['task_{0}'.format(i)], self.data_f['task_{0}'.format(j)]))
                else:
                    self.CovMat['task_{0}{1}'.format(i, j)] = np.transpose(self.CovMat['task_{0}{1}'.format(j, i)])
        
        # To join all the covariance matrices defined above in the dictionaries into one big covariance matrix taking into account the cross correlations between the datasets 
        self.CovMat_all = np.block([[ self.CovMat['task_{0}{1}'.format(i, j)] for i in range(self.ntasks)] for j in range(self.ntasks)])  
        
        return self.CovMat_all
    
    def derivative_kernel(self, model1, model2, params1, params2, x1, x2=None):
        """ 
            Defines the derivatives of cross kernel function for the GP model and returns a matrix 
            """
        dK = {}
        if x2 is None:
            x2 = x1
        if model1 == 'SE' and model2 == 'SE':
            # product of the sigmas
            sig1sig2 = params1[0] * params2[0]
            # product of the scale lengths
            l1l2 = params1[1] * params2[1]
            # quadrature of length scales
            l1l2_quad = params1[1]**2. + params2[1]**2.
            # difference between x1 and x2
            x1x2 = x1[:, None]-x2[None, :]

            # derivativve wrt x1 and derivative wrt x2
            dK['dK_dx1'] = - 2 * sig1sig2 * x1x2 * np.exp(- (x1x2**2.)/(l1l2_quad)) * np.sqrt(2. * l1l2) / np.sqrt(l1l2_quad**3.) 
            dK['dK_dx2'] = - dK['dK_dx1']
            # derivative wrt x1 and x2
            dK['d2K_dx1dx2'] = 2 * sig1sig2 * (1. - 2 * (x1x2**2.)/(l1l2_quad)) * np.exp(- (x1x2**2.)/(l1l2_quad)) * np.sqrt(2. * l1l2) / np.sqrt(l1l2_quad**3.)
            # double derivative wrt x1
            dK['d2K_dx1dx1'] = 2 * sig1sig2 * (2. * (x1x2**2.)/(l1l2_quad) - 1.) * np.exp(- (x1x2**2.)/(l1l2_quad)) * np.sqrt(2. * l1l2) / np.sqrt(l1l2_quad**3.)


            
            # # Double derivative wrt x1 and x2
            # dK['d2K_dx1dx2'] = params1[0] * params2[0] * (1. - ((x1[:, None]-x2[None, :])**2.)/(params1[1]**2. + params2[1]**2.)) * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(params1[1]**2. + params2[1]**2.)) * np.sqrt(2. * params1[1] * params2[1] / (params1[1]**2. + params2[1]**2.))
            # # Double derivative wrt x1 and double derivative wrt x2
            # dK['d2K_dx1dx1'] = params1[0] * params2[0] * (params2[1]**2. - (x1[:, None]-x2[None, :])**2.) * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(params1[1]**2. + params2[1]**2.)) * np.sqrt(2. * params1[1] * params2[1] / (params1[1]**2. + params2[1]**2.))
            # dK['d2K_dx2dx2'] = dK['d2K_dx1dx1']
            # # Double derivative wrt x1 and derivative wrt x2
            # dK['d3K_dx1dx1dx2'] = - params1[0] * params2[0] * (x1[:, None]-x2[None, :]) * (params2[1]**2. - (x1[:, None]-x2[None, :])**2.) * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(params1[1]**2. + params2[1]**2.)) * np.sqrt(2. * params1[1] * params2[1] / (params1[1]**2. + params2[1]**2.))
            # # Double derivative wrt x1 and x2
            # dK['d3K_dx1dx2dx2'] = - params1[0] * params2[0] * (x1[:, None]-x2[None, :]) * (params1[1]**2. - (x1[:, None]-x2[None, :])**2.) * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(params1[1]**2. + params2[1]**2.)) * np.sqrt(2. * params1[1] * params2[1] / (params1[1]**2. + params2[1]**2.))
            # # Triple derivative wrt x1 and x2
            # dK['d4K_dx1dx1dx2dx2'] = params1[0] * params2[0] * (params1[1]**2. - (x1[:, None]-x2[None, :])**2.) * (params2[1]**2. - (x1[:, None]-x2[None, :])**2.) * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(params1[1]**2. + params2[1]**2.)) * np.sqrt(2. * params1[1] * params2[1] / (params1[1]**2. + params2[1]**2.))

            return dK
            
    