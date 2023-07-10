import sys
import os

import cmath
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

import time
import configparser

from multiprocessing import Pool
import itertools
import pickle

from marcia import Data
from marcia import GPconfig
from marcia.backend import kernel as kernel

class Kernels(object):
    """
        General Gaussian process modelling that all the specific kernels must be inherited from.
        Inputs would be the appropriate amplitude and scaling parameters needed for the particular model. 
        This initialisation retuns the final covarinace matrices as functions of kernal paramters. 
        """
    def __init__(self, data, filename = None, verbose = False):
        """
        Initialisation takes the data and the filename for the config file, which declares all the GP 
        configurations.
        """
        self.clight = 299792458. / 1000.  # km/s
        # Read the config file
        self.ini_path = os.path.join(os.path.dirname(__file__), 'GPconfig.ini')
        print(f'Loading the config file from {self.ini_path} ... ')
        MTGP = GPconfig.GPConfig(filename if filename else self.ini_path).__dict__
        
        # Read the models and the number of tasks
        self.models = MTGP['models']
        self.nTasks = MTGP['n_Tasks']
        self.nmodel = self.nTasks

        # Read the nus from the config file
        self.nus = MTGP['nus']

        # The data contains the list/lists of x-axes
        self.data = data
        self.ndata = len(data)

        # Print details of the run when the class is initialised
        print(f'Number of tasks: {self.nTasks}')
        print(f'Number of datasets: {len(self.data)}')
        print(f'Kernels: {self.models}')

        self.setup_kernel_data()
        
        # To check if the  number of datasets is equal to the number of models
        if len(self.data) != self.nmodel:
            raise ValueError(f'Error: data length is {len(self.data)} and does not match the number of kernels {self.nmodel}')
        
        if len(set(self.models)) == 1 and self.nTasks > 1 and self.models[0] == 'SE':
            print(f'Perfroming a multi task Gaussian process regression using {self.models[0]} kernels and {self.nTasks} tasks')
            pass
        elif len(set(self.models)) == 1 and self.nTasks > 1 and self.models[0] in ['M92', 'M72', 'M52', 'M32']:
            # if all the kernels are the same, we can define the cross covariance matrix
            print(f'Perfroming a multi task Gaussian process regression using {self.models[0]} kernels and {self.nTasks} tasks')
            # To check if the interpolated cross covariance matrix exists in the backend folder

            self.file_path = os.path.join(os.path.dirname(__file__), f'backend/cross_kernels/KcMM_{self.models[0]}_{self.models[0]}.pck')
            print(f'Checking if the cross covariance matrix exists in {self.file_path} ... ')
            if not os.path.exists(self.file_path):
                # Here we update any necessary interpolation parameters
                self.nu1 = self.nu2 = self.nus[0]
                self.make_cross_kernel()
            
            print(f'Loading the cross covariance matrix for the multi task GP model with the {self.models[0]} kernel ... ')
            with open(self.file_path, 'rb') as f:
                self.KcMM_int = pickle.load(f)
        elif len(set(self.models)) == 1 and self.nTasks==1:
            print(f'Perfroming a simple Gaussian process regression for a single task using {self.models[0]} kernel')
        else:
            # Cross covariance is not defined for different kernels
            raise ValueError('Error: Multi kernel cross covariance matrix is not defined, only multi task for same kernel is defined')
        
        # To initialise the total covariance matrix as a dictionary which is a function of the kernel parameters
        self.CovMat = {}

    def setup_kernel_data(self):
        """Define kernels and data in a dictionary."""
        self.kernel_f = {}
        self.data_f = {}
        for i in range(self.nTasks):
            self.kernel_f[f't_{i}'] = self.models[i]
            self.data_f[f't_{i}'] = self.data[i]

        # Define the data_tau dictionary, which contains the difference between the x-axes of the datasets
        self.data_tau = {}
        for i in range(self.nTasks):
            for j in range(self.nTasks):
                self.data_tau[f't_{i}{j}'] = self.data_f[f't_{i}'][:, None] - self.data_f[f't_{j}'][None, :]
        
    def __call__(self, pars):
        """
            It returns the covariance matrix for the GP model for the given paramters 
        """
        # We set the self-scaling in the likelihood function and here simply create the dictionary of parameters
        # self.CovMat_all = np.block([[ pars[i,0]* pars[j,0]*self.CovMat[f't_{i}{j}']([pars[i,1]* pars[j,1], self.data_tau[f't_{i}{j}']]) for i in range(self.nTasks)] for j in range(self.nTasks)])

        self.CovMat_all = self.CovMat_all(pars)
        return self.CovMat_all
        
    # Define the special fucntions needed for the generalised matern kernel
    def _spkv_(self, tau, l1):
        return sp.special.kv(2., 3. * np.abs(tau) / l1)
    
    def _spgamma_(self, x):
        return sp.special.gamma(x)

    def gMdef(self, tau, nu, l1):
        """ 
            Defines the basis function for the GP model and returns a vector
            Finally I define only ine single function for the basis function
            This is independent of the amplitude paramter sigma_f
            """
        x = np.abs(tau)
        if nu == 0.0: # Squared Exponential kernel
            return np.exp(- (x**2.)/(2. * l1**2.))
        else: # Generalised Matern kernel
            A, B, C = kernel.gMdef(tau,l1,nu)
            A = A * (self._spgamma_(nu/2. - 1./4.)/self._spgamma_(nu/2. + 1./4.))**(1./2.) * (self._spgamma_(nu + 1./2.)/self._spgamma_(nu))**(1./4.)
            B = B / self._spgamma_(nu/2. - 1./4.)
            return A**2. * B * C**(nu/2. - 1./4.) * self._spkv_(tau,l1) / np.pi

        
    def gdMdef(self, tau, nu, l1):
        """
            Defines the derivative of the basis function for the GP model and returns a vector
            Finally I define only ine single function for the basis function"""
        B = sp.special.kv(1./4. * (5. - 2. * nu), np.sqrt(2.) * np.sqrt(nu) * np.sqrt(tau**2.) / l1)
        C = np.sqrt(self._spgamma_(1./2. + nu) / self._spgamma_(nu))
        A, D = kernel.gdMdef(tau, l1, nu)
        D = D * self._spgamma_(1./4. + nu/2.) 
        return A * B * C / D
    
    def gMMdef_integrand(self, params):
        tau, l1, l2 = params
        gMMdef_integrand = lambda u: self.gMdef(u, self.nu1, l1) * self.gMdef(tau - u, self.nu2, l2)
        integral, _ = quad(gMMdef_integrand, -np.inf, np.inf, epsabs=1e-8, epsrel=1e-8, limit=100)
        return integral
    

    def KcMM(self, l1_values, l2_values, tau_values):
        """
            Defines the cross kernel function for the GP model and returns the covariance matrix
            """
        integrals = np.zeros((len(tau_values), len(l2_values), len(l1_values)))
        params = list(itertools.product(tau_values, l2_values, l1_values))
        with Pool() as pool:
            integrals = pool.map(self.gMMdef_integrand, params)
        integrals = np.array(integrals).reshape((len(tau_values), len(l2_values), len(l1_values)))
        return RegularGridInterpolator((tau_values, l2_values, l1_values), integrals, bounds_error=False, fill_value=None)

    def matern(self, nu, x1, x2, l_s):
        """
            Defines the generalised matern kernel https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
            nu is a positive half-integer that determines the differentiability of the function and the smoothness of the resulting Gaussian process sample paths.
            nu = 1/2, 3/2, 5/2, ... tend to be used most often in practice.
            nu tends to infinity, the matern kernel tends to the squared exponential kernel
            nu tends to 0, the matern kernel tends to the absolute exponential kernel
            The input x1 and x2 are the x-axes of the data points and l_s is the scale length
            """
        # Here to-do the nu = half-integer cases -- Sandeep
        # This fucntion returns nan when x1 = x2, so we need to replace nan with zero
        return np.nan_to_num((2.**(1.-nu)/sp.special.gamma(nu)) * ((np.sqrt(2.*nu) * np.abs(x1-x2))/l_s)**nu * sp.special.kv(nu, (np.sqrt(2.*nu) * np.abs(x1-x2))/l_s)) + np.eye(len(x1)) 

    def kernel(self, model, params, x1, x2=None, mat_nu=None):
        """ 
            Defines the kernel function for the GP model and returns the covariance matrix 
            """
        if x2 is None:
            x2 = x1
        if model == 'SE': # Squared Exponential kernel
            return params[0]**2. * np.exp(- ((x1[:, None]-x2[None, :])**2.)/(2. * params[1]**2.))
        if model in ['M92', 'M72', 'M52', 'M32']: # Matern 9/2 kernel
            return params[0]**2. * self.matern(mat_nu, x1[:, None], x2[None, :], params[1])
        if model == 'matern' and mat_nu is not None:
            return params[0]**2. * self.matern(mat_nu, x1[:, None], x2[None, :], params[1])

    def cross_kernel(self, model1, model2, params1, params2, x1, x2):
        """ 
            Defines the cross kernel function for the GP model and returns the covariance matrix
            """
        if model1 == model2:
            model_here = model1
            # difference between x1 and x2
            x1x2 = x1[:, None]-x2[None, :]
            # product of the sigmas
            sig1sig2 = params1[0] * params2[0]

            if model_here == 'SE': # Squared Exponential kernel
                # product of the scale lengths
                l1l2 = params1[1] * params2[1]
                # quadrature of length scales
                l1l2_quad = params1[1]**2. + params2[1]**2.
                # This fucntion boils down to the SE kernel function when the scale lengths are equal for both the datasets
                return sig1sig2 * np.exp(- (x1x2**2.)/(l1l2_quad)) * np.sqrt(2. * l1l2 / l1l2_quad)
            elif model_here in ['M92', 'M72', 'M52', 'M32']: 
                # the pickle file name is the 'KcMM_M92_M92.pkl' where the first M92 is the model1 and the second M92 is the model2
                lambda_func = np.vectorize(lambda x: self.KcMM_int([x, params1[1], params2[1]]))
                return sig1sig2 * np.array(lambda_func(x1x2))
                
        else:
            raise ValueError(f'Error: Cross kernel not defined for {model1} and {model2}')


    def Cov_Mat(self, params):
        """ 
            Defines the covariance matrix for the GP model and returns the covariance matrix 
            """
        # To create a dictionary of covariance matrices for each task
        
        for i in range(self.nTasks):
            for j in range(self.nTasks):
                if i == j:
                    self.CovMat[f't_{i}{j}'] = self.kernel(self.kernel_f[f't_{i}'], params[i], self.data_f[f't_{i}'], mat_nu=self.nus[i])
                elif i < j:
                    self.CovMat[f't_{i}{j}'] = np.transpose(self.cross_kernel(self.kernel_f[f't_{i}'], self.kernel_f[f't_{j}'], params[i], params[j], self.data_f[f't_{i}'], self.data_f[f't_{j}']))
                else:
                    self.CovMat[f't_{i}{j}'] = np.transpose(self.CovMat[f't_{j}{i}'])
        
        # To join all the covariance matrices defined above in the dictionaries into one big covariance matrix taking into account the cross correlations between the datasets 
        CovMat_all = np.block([[ self.CovMat[f't_{i}{j}'] for i in range(self.nTasks)] for j in range(self.nTasks)])
        
        return CovMat_all
   
    def make_cross_kernel(self):
        """This function is used to create and save the interpolated cross kernel for the multi task GP model"""
        
        print(f'Constructing the cross covariance matrix for the multi task GP model with the {self.models[0]} kernel ... ')
        self.nu1 = self.nu2 = self.nus[0]
        # The following has to be changed to modify the resolution of the cross covariance matrix
        l1_values = np.linspace(0.01, 20, 100)
        l2_values = np.linspace(0.01, 20, 100)
        tau_values = np.linspace(-10, 10, 100)

        integrals = np.zeros((len(tau_values), len(l2_values), len(l1_values)))
        params = list(itertools.product(tau_values, l2_values, l1_values))
        with Pool() as pool:
            integrals = pool.map(self.gMMdef_integrand, params)
        integrals = np.array(integrals).reshape((len(tau_values), len(l2_values), len(l1_values)))


        KcMM_int = RegularGridInterpolator((tau_values, l2_values, l1_values), integrals, bounds_error=False, fill_value=None)
        with open(self.file_path, 'wb') as f:
            pickle.dump(KcMM_int, f)
        print(f'Cross covariance matrix saved in {self.file_path}')


    # class predict(Kernels):
    #     """
    #         This class is used to predict the mean and the variance of the GP model for a given set of optimized hyper-parameters
    #         """
    #     # import attributes from the parent class Kernels with out the need to initialise the parent class
    #     def __init__(self, data_x, data_y, data_cov, target, filename = None, verbose = False):
    #         self.__dict__ = Kernels(data_x, filename = None, verbose = False).__dict__
    #         self.data_x = data_x
    #         self.data_y = data_y
    #         self.data_cov = data_cov
    #         self.target = target
    #         self.setup_kernel_data()

    #         if len(self.target) != self.nTasks:
    #             raise ValueError(f'Error: target length is {len(self.target)} and does not match the number of kernels {self.nTasks}')

    #     def __call__(self, params):
    #         """
    #             It returns the mean and the variance of the GP model for the given paramters
    #             """
    #         # The total covariance is sum of the covariance matrix and the data covariance matrix
    #         self.CovMat_all = self.Cov_Mat(params) + self.data_cov
    #         self.reconstructed = self.reconstruct(params)
        
    #         return self.reconstructed
            
    #     def reconstruct(self, params):
    #         """
    #             It returns the reconstructed funcitons for the given paramters at the target
    #             """
    #         # To reconstruct each task separately
    #         for i in range(self.nTasks):
    #             # construcnt the covariance matrix for each task at the target
    #             self.CovMat_task = self.kernel(self.kernel_f[f't_{i}'], params[i], self.data_f[f't_{i}'], self.target, mat_nu=self.nus[i])
    #             mean = np.dot(self.CovMat_task, np.linalg.solve(self.CovMat_all, self.target))

            
    #         variance = np.diag(self.CovMat_all) - np.diag(np.dot(self.CovMat_all, np.linalg.solve(self.CovMat_all, np.transpose(self.CovMat_all))))
    #         return mean, variance
