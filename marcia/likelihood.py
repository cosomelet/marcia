import numpy as np
from marcia import Data 
from marcia import Cosmology as cosmo 


class Likelihood(object):
    # This contains the likelihoods for all the datasets implemented in the code as yet
    def __init__(self, model,parameters,data,prior_file=None):

        self.clight = 299792458. / 1000.
        self.theory = cosmo(model,parameters,prior_file) # This needs to be done here to make the code more autonomous
        # This list sets the available data to work with, has to be updated when ever a new dataset is added
        self.prior = self.theory.priors
        self.data = data
        # The nuisance parameters for each of the datasets, if any have to be added here to the theta
        

    def __getattribute__(self, item):
        return super(Likelihood, self).__getattribute__(item)

            
    def chisq_CC(self,theta):
        data = Data(data)
        hubble_theory = self.theory.hubble_rate(theta)
        redshift, hubble_rate, covariance = data.get_cosmic_clocks()
        return  np.dot( data.cc , np.dot( theory.CovMat , data.cc.T) ) + np.log( theory.DetCovMat ) +  np.log(2. *np.pi )*len(data.z)


    def chisq(self,theta):
        chi2 = 0
        for i in self.data:
            chi2 += self.__getattribute__(f'chisq_{i}')(theta)
        return chi2
    
    def logPrior(self, theta):
        # if self.priors[0][0] < theta[0] < self.priors[0][1] and self.priors[1][0] < theta[1] < self.priors[1][1]:
        # logPrior is independent of data for the most of it, unless otherwise some strange functions are defined
        if all((np.array(theta)-self.priors[:, 0])>0) and all((self.priors[:, 1]-np.array(theta))>0):
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
