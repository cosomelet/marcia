import numpy as np
from marcia import Data 
from marcia import Cosmology as cosmo 


class Likelihood(object):
    # This contains the likelihoods for all the datasets implemented in the code as yet
    def __init__(self, model,parameters,data,prior_file=None):

        self.clight = 299792458. / 1000.
        self.theory = cosmo(model,parameters,prior_file) # This needs to be done here to make the code more autonomous
        # This list sets the available data to work with, has to be updated when ever a new dataset is added
        self.priors = self.theory.priors
        self.data = data
        self.db = Data(data)
        # The nuisance parameters for each of the datasets, if any have to be added here to the theta
        

    #def __getattribute__(self, item):
    #    return super(Likelihood, self).__getattribute__(item)

            
    def chisq_CC(self,theta):
        redshift, hubble_rate, covariance = self.db.get_cosmic_clocks()
        hubble_theory = self.theory.hubble_rate(theta, redshift)
        return  np.dot(hubble_rate - hubble_theory, np.dot(np.linalg.inv(covariance), hubble_rate - hubble_theory))
    
    def chisq_BAO_alam(self,theta):
        redshift, distance, covariance = self.db.get_BAO_alam()
        rd = self.theory.sound_horizon(theta)
        dist = distance.copy()
        distance_theory = self.theory.transverse_distance(theta, redshift)
        dist*=rd
        cov = covariance.copy()
        cov*=rd**2

        return  np.dot(dist- distance_theory, np.dot(np.linalg.inv(cov), dist - distance_theory))


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
