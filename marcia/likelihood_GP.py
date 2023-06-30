import numpy as np
from marcia import Data 
from marcia import GPconfig
from marcia import CosmoConfig
from marcia import Cosmology as cosmo 
from marcia import kernels as kern


class Likelihood_GP(object):
    # This contains the generalized likelihood for the GP and MTGP
    def __init__(self, data, GP_filename=None, cosmo_filename=None):
        
        # Read form the GP config file
        # read the config file
        if GP_filename is None:
            MTGP = GP_filename.GPConfig('GPconfig.ini').__dict__
        else:
            MTGP = GP_filename.GPConfig(GP_filename).__dict__

        # To read the cosmology file and inculde a comological mean function in the GP
        if cosmo_filename is None:
            cosmo = cosmo_filename.CosmoConfig('cosmology.ini').__dict__
        else:
            cosmo = cosmo_filename.CosmoConfig(cosmo_filename).__dict__

        # To include the cosmological mean function in the GP
        if cosmo['sample'] == False:
            self.cosmo = cosmo['model']
        elif cosmo['sample'] == True:
            # Yet to define the cosmological mean function for the sampled cosmology with paramters sampled amongside the GP hyperparameters
            pass
        else:
            self.mean = [0.0] * len(np.flatten(data.x.values()) )

        # read the models and the number of tasks
        self.models = MTGP['models']
        self.nTasks = MTGP['n_Tasks']
        self.nmodel = self.nTasks
        self.self_scale = MTGP['self_scale']
        self.scatter = MTGP['INTRINSIC_SCATTER']['sigma_int']

        # To define the priors for the GP hyperparameters using the the GPconfig file
        self.priors = []
        for i in range(self.nTasks):
            self.priors.append([MTGP[f'Task_{i}']['sigma_f_min'], MTGP[f'Task_{i}']['sigma_f_max']])
            self.priors.append([MTGP[f'Task_{i}']['l_min'], MTGP[f'Task_{i}']['l_max']])
            
        self.priors = np.array(self.priors)
        
        # validate the number of models and data 
        if len(self.models) != len(data):
            raise ValueError('The number of models and the number of data are not equal')
        
        # read the data and set the data covarince matrix
        self.d = Data(data)
        self.x, self.y, self.cov = self.d()
        self.D_covmat = self.cov

        # intialize the kernals from Kernel class
        self.GP_kernels = kern.Kernels(list(data.x.values()))

    def set_theta(self, theta):
        # To reorganise the theta values for the GP hyperparameters in accordanace with the GPconfig file
        # theta contains no self-scale: [sigma_f_1, l_1, sigma_f_2, l_2, ...] 
        # or [sigma_f_1, sigma_f_2, ..., l] if self_scale is True 
        # and possibly with intrinsic scatter at the end [..., sigma_int] 

        if self.self_scale == False and len(theta) != 2*self.nTasks and self.scatter == False:
            raise ValueError('The number of hyperparameters does not match the number of tasks')
        elif self.self_scale == False and len(theta) != 2*self.nTasks+1 and self.scatter == True:

            params = []
            for i in range(self.nTasks):
                params.append(theta[2*i])
                params.append(theta[2*i+1])
            params = np.array(params)

        return params


    def logPrior(self, theta):
        if all(self.priors[i][0] < theta[i] < self.priors[i][1] for i in range(len(theta))):
            return 0.0
        return -np.inf

    def logLike(self,theta):
        chi2 = self.chisq(theta)
        return -0.5*chi2

    def logProb(self, theta):
        lp = self.logPrior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.logLike(theta)
    
    def chisq(self,theta):
        # We define the GP likelihood here
        chi2 = 0
        # To call the GP class to get the covariance matrix
        GP_covmat = self.GP_kernels(theta)
        Total_covmat = GP_covmat + self.D_covmat

        # To perform cholesky decomposition
        L = np.linalg.cholesky(Total_covmat)
        L_inv = np.linalg.inv(L)
        y = self.y - self.mean
        y_inv = np.dot(L_inv, y)

        # To calculate the log determinant
        log_det = 2*np.sum(np.log(np.diag(L)))

        # To calculate the chi2
        chi2 = np.dot(y_inv.T, y) + log_det

        return chi2


        



