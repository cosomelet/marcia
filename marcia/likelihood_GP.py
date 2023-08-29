import sys
import numpy as np
from marcia import Data 
from marcia import GPconfig
from marcia import CosmoConfig
from marcia import Cosmology as cosmo 
from marcia import Kernel as kern


class Likelihood_GP(object):
    # This contains the generalized likelihood for the GP and MTGP
    def __init__(self, data, GP_filename=None, cosmo_filename=None):
        
        # Read form the GP config file
        MTGP = GPconfig.GPConfig(GP_filename if GP_filename else 'GPconfig.ini').__dict__

        # read the models and the number of tasks
        self.models = MTGP['models']
        self.nTasks = MTGP['n_Tasks']
        self.nmodel = self.nTasks
        self.self_scale = MTGP['self_scale']
        self.scatter = MTGP['sigma_int']

        # To define the priors for the GP hyperparameters using the the GPconfig file
        self.priors = []
        for i in range(self.nTasks):
            i = i+1 # To start from 1
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

        # To set the mean function 
        self.mean = [0.0]*len(self.y)
        # if MTGP['cosmo_model'] == None:
        #     self.mean = [0.0]*len(self.y)
        # else:
        #     # To read the cosmology config file
        #     raise NotImplementedError('Cosmology mean is not implemented yet')

        # intialize the kernals from Kernel class
        self.GP_kernels = kern(list(self.d.x.values()), filename=GP_filename)

    def set_theta(self, theta):
        # To reorganise the theta values for the GP hyperparameters in accordanace with the GPconfig file
        # theta contains no self-scale: [sigma_f_1, l_1, sigma_f_2, l_2, ...]
        # or [sigma_f_1, sigma_f_2, ..., l] if self_scale is True
        # and possibly with intrinsic scatter at the end [..., sigma_int]
        # Final form is [[sigma_f_1, l_1], [sigma_f_2, l_2], ...]

        if self.self_scale == False and len(theta) != 2*self.nTasks and self.scatter == False:
            raise ValueError('The number of hyperparameters does not match the number of tasks')
        elif self.self_scale == False and len(theta) != 2*self.nTasks+1 and self.scatter == True:
            pass
        else:
            # rearrange the theta values into a list of lists
            params = []
            for i in range(self.nTasks):
                params.append([theta[2*i], theta[2*i+1]])

        return params

    def logPrior(self, theta):
        # To set the priors for the GP hyperparameters
        for (lower, upper), value in zip(self.priors, theta):
            if not lower < value < upper:
                # print(value, lower, upper)
                # sys.exit()
                return -np.inf
        return 0.0
    
    def logLike(self,theta):
        chi2 = self.chisq(theta)
        return -0.5*chi2

    def logProb(self, theta):
        lp = self.logPrior(theta)
        theta = self.set_theta(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.logLike(theta)
    
    def chisq(self,theta):
        # We define the GP likelihood here
        chi2 = 0
        # To call the GP class to get the covariance matrix
        GP_covmat = self.GP_kernels.Cov_Mat(theta)
        Total_covmat = GP_covmat + self.D_covmat + 1e-6*np.eye(len(self.D_covmat))

        # To perform cholesky decomposition
        L_inv = np.linalg.inv(Total_covmat)
        y = self.y - self.mean
    
        # To calculate the log determinant
        log_det = np.linalg.slogdet(Total_covmat)[1]
        offset = len(y)*np.log(2*np.pi)

        # To calculate the chi2
        y_inv = np.dot(L_inv, y)
        chi2 = np.dot(y, y_inv.T) + log_det - offset

        return chi2


        



