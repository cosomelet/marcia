import emcee
import numpy as np
from marcia import Likelihood as lk
import scipy.optimize as op
from chainconsumer import ChainConsumer

class Sampler:

    def __init__(self,model,parameters,data,initial_guess,prior_file=None):
        self.likelihood = lk(model,parameters,data,prior_file)

        self.ndim = len(self.likelihood.priors)
        self.nwalkers = 100
        self.nsteps = 20
        self.pburnin = 20. # percentage of burn-in steps to remove from each walker
        self.nburnin =self.nsteps*(self.pburnin)/100
        self.initial_guess = initial_guess

    
    def MLE(self,verbose=True):
        nll = lambda x: -self.likelihood.logProb(x)
        result = op.minimize(nll, x0=self.initial_guess, method = 'Nelder-Mead', options={'maxfev': None})
        if verbose:
            print(f'Best-fit values: {result.x}')
            print(f'Max-Likelihood value (including prior likelihood):{self.likelihood.logProb(result.x)}')
        return result.x
    

    def sampler(self):
        mle = self.MLE()
        pos = [mle+ 1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood.logProb)
        sampler.run_mcmc(pos, self.nsteps,progress=True)
        return sampler
    
    def get_chain(self):
        sampler = self.sampler()
        samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return samples

    def corner_plot(self):
        chains = self.get_chain()
        c = ChainConsumer().add_chain(chains, parameters=self.likelihood.theory.labels)
        fig = c.plotter.plot(truth=list(self.MLE(False)))
        fig.set_size_inches(3 + fig.get_size_inches()) 
        return fig


    
