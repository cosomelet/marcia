import emcee
import numpy as np
from marcia import Likelihood as lk
import scipy.optimize as op

class Sampler:

    def __init__(self,model,parameters,data,prior_file=None):
        self.likelihood = lk(model,parameters,data,prior_file)

        self.ndim = len(self.likelihood.priors)
        self.nwalkers = 100
        self.nsteps = 500
        self.pburnin = 30. # percentage of burn-in steps to remove from each walker
        self.nburnin =self.nsteps*(self.pburnin)/100

    
    def MLE(self,initial_guess):
        nll = lambda x: -self.likelihood.logProb(x)
        result = op.minimize(nll, x0=initial_guess, method = 'Nelder-Mead', options={'maxfev': None})
        print(f'Best-fit values: {result.x}')
        print(f'Max-Likelihood value (including prior likelihood):{self.likelihood.logProb(result.x)}')
        return result.x