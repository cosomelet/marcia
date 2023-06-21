import numpy as np
from marcia import Data 
from marcia import GPconfig
from marcia import Cosmology as cosmo 
from marcia import kernels as kern


class Likelihood_GP(object):
    # This contains the generalized likelihood for the GP and MTGP
    def __init__(self, model,parameters,data,GP_filename=None, cosmo_filename=None):
        
        # Read form the GP config file
        # read the config file
        if GP_filename is None:
            MTGP = GP_filename.GPConfig('GPconfig.ini').__dict__
        else:
            MTGP = GP_filename.GPConfig(GP_filename).__dict__
        
        # read the models and the number of tasks
        self.models = MTGP['models']
        self.nTasks = MTGP['n_Tasks']
        self.nmodel = self.nTasks
        self.self_scale = MTGP['self_scale']
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


    def logPrior(self, theta):
        # if self.priors[0][0] < theta[0] < self.priors[0][1] and self.priors[1][0] < theta[1] < self.priors[1][1]:
        # logPrior is independent of data for the most of it, unless otherwise some strange functions are defined
        #if all((np.array(theta)-self.priors[:, 0])>0) and all((self.priors[:, 1]-np.array(theta))>0):
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
        y_inv = np.dot(L_inv, self.y)

        # To calculate the log determinant
        log_det = 2*np.sum(np.log(np.diag(L)))

        # To calculate the chi2
        chi2 = np.dot(y_inv.T, y_inv) + log_det

        return chi2


        



