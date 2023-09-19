import numpy as np
from marcia import Data 
from marcia import Cosmology as cosmo 

import sys
from marcia import GPconfig
from marcia import CosmoConfig
from marcia.kernel import Kernel as kern
from marcia import Params
import pdb


class Likelihood(object):
    # This contains the likelihoods for all the datasets implemented in the code as yet
    def __init__(self, model,parameters,data,prior_file=None):

        self.clight = 299792458. / 1000.
        self.theory = cosmo(model,parameters,prior_file) # This needs to be done here to make the code more autonomous
        # This list sets the available data to work with, has to be updated when ever a new dataset is added
        self.priors = self.theory.priors
        self.params = Params(parameters)
        self.data = data
        self.db = Data(data)
        self.inv_covariance = {}
        # The nuisance parameters for each of the datasets, if any have to be added here to the theta
        

    #def __getattribute__(self, item):
    #    return super(Likelihood, self).__getattribute__(item)

            
    def chisq_CC(self,theta):
        redshift, hubble_rate, covariance = self.db.get_cosmic_clocks()
        hubble_theory = self.theory.hubble_rate(theta, redshift)
        if 'CC' in self.inv_covariance.keys():
            icov = self.inv_covariance['CC']
        else:
            icov = np.linalg.inv(covariance)
            self.inv_covariance['CC'] = icov
        return  np.dot(hubble_rate - hubble_theory, np.dot(icov, hubble_rate - hubble_theory))
    
    def chisq_BAO_alam(self,theta):
        redshift, distance, covariance = self.db.get_BAO_alam()
        rd = self.theory.sound_horizon(theta)
        dist = distance.copy()
        distance_theory = self.theory.transverse_distance(theta, redshift)
        dist*=rd
        cov = covariance.copy()
        cov*=rd**2
        if 'BAO_alam' in self.inv_covariance.keys():
            icov = self.inv_covariance['BAO_alam']
        else:
            icov = np.linalg.inv(cov)
            self.inv_covariance['BAO_alam'] = icov

        return  np.dot(dist- distance_theory, np.dot(np.linalg.inv(cov), dist - distance_theory))
    
    def chisq_Pantheon_plus(self,theta):
        cmb_z, mb, covariance = self.db.get_pantheon_plus()
        helio_z = self.db.get_pantheon_plus(Zhel=True)
        distance_theory = self.theory.distance_modulus(theta, cmb_z, helio_z)
        delta = mb - distance_theory
        if 'Pantheon_plus' in self.inv_covariance.keys():
            icov = self.inv_covariance['Pantheon_plus']
        else:
            icov = np.linalg.inv(covariance)
            self.inv_covariance['Pantheon_plus'] = icov

        return np.dot(delta, np.dot(icov, delta))

    def chisq_Pantheon_old(self,theta):
        cmb_z, mb, covariance = self.db.get_pantheon_old()
        helio_z = self.db.get_pantheon_old(Zhel=True)
        distance_theory = self.theory.distance_modulus(theta, cmb_z, helio_z)
        delta = mb - distance_theory
        if 'Pantheon_old' in self.inv_covariance.keys():
            icov = self.inv_covariance['Pantheon_old']
        else:
            icov = np.linalg.inv(covariance)
            self.inv_covariance['Pantheon_old'] = icov

        return np.dot(delta, np.dot(icov, delta))
    
    def chisq_QSO_dm(self,theta):
        p = self.params(theta)
        #pdb.set_trace()
        z,dm,cov = self.db.get_QSO()
        distance_theory = self.theory.distance_modulus(theta, z,z)
        delta = dm - distance_theory
        var = np.diag(cov)
        return np.sum(delta**2./(var + p.qso_sigma**2))
    
    def chisq_QSO_full(self,theta):
        p = self.params(theta)
        z, lnFUV, lnFUV_err, lnFX, lnFX_err = self.db.get_QSO_full()
        yi = lnFX
        dyi = lnFX_err
        xi = lnFUV
        dxi = lnFUV_err
        DL = self.theory.luminosity_distance(theta,z)*3.086e24
        psi = p.qso_beta + p.qso_gamma*(xi) + 2*(p.qso_gamma -1)*(np.log10(DL)) + (p.qso_gamma-1)*np.log10(4*np.pi)

        si2 = p.qso_gamma**2 * dxi**2 + dyi**2 + np.exp( 2*np.log(p.qso_sigma) )

        return np.sum((psi - yi)**2/si2 + np.log(2*np.pi*si2))



    def chisq(self,theta):
        chi2 = 0
        for i in self.data:
            i = i.replace('-','_')
            chi2 += self.__getattribute__(f'chisq_{i}')(theta)
        return chi2
    
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


