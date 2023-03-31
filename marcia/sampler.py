import emcee
import numpy as np
from marcia import Likelihood as lk
import scipy.optimize as op

class Sampler:

    def __init__(self,data,model,priors):
        self.data = data
        self.model = model
        self.priors = priors
        self.likelihood = lk(data,model,priors)

        self.ndim = len(self.priors)
        self.nwalkers = 100
        self.nsteps = 500
        self.pburnin = 30. # percentage of burn-in steps to remove from each walker
        self.nburnin =self.nsteps*(self.pburnin)/100

    
    def MLE(self):
        nll = lambda x: -self.likelihood.logProb(x)
        result = op.minimize(nll, x0=self.priors, method = 'Nelder-Mead', options={'maxfev': None})
        print(f'Best-fit values: {result.x}')
        print(f'Max-Likelihood value (including prior likelihood):{self.likelihood.logProb(result.x)}')
        return result.x