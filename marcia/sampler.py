import emcee
import numpy as np
from marcia import Likelihood as lk
import scipy.optimize as op
from chainconsumer import ChainConsumer
import logging
import os

logging.basicConfig(filename="convergence.log",format='%(asctime)s %(message)s',filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)



class Sampler:

    def __init__(self,model,parameters,data,initial_guess,prior_file=None):
        self.likelihood = lk(model,parameters,data,prior_file)

        self.ndim = len(self.likelihood.priors)
        self.nwalkers = 100
        self.initial_guess = initial_guess

    
    def MLE(self,verbose=True):
        nll = lambda x: -self.likelihood.logProb(x)
        result = op.minimize(nll, x0=self.initial_guess, method = 'Nelder-Mead', options={'maxfev': None})
        if verbose:
            print(f'Best-fit values: {result.x}')
            print(f'Max-Likelihood value (including prior likelihood):{self.likelihood.logProb(result.x)}')
        return result.x
    

    def sampler_pos(self):
        mle = self.MLE()
        pos = [mle+ 1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        return pos
    
    def sampler(self,filename):
        backend = emcee.backends.HDFBackend(filename)
        if os.path.isfile(filename):
            return backend
        else:

            backend.reset(self.nwalkers, self.ndim)

            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood.logProb, backend=backend)  

            max_n = 100000
            index = 0
            autocorr = np.empty(max_n)

            old_tau = np.inf

            for sample in sampler.sample(self.sampler_pos(), iterations=max_n, progress=True):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                logger.info(f'I: {sampler.iteration}, tau*100:{tau * 100}, Tol:{(old_tau - tau) / tau}')
                if converged:
                    print(f'Converged at iteration {sampler.iteration}')
                    break
                old_tau = tau
            
            return sampler

# Initialize the sampler


        # sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood.logProb)
        # sampler.run_mcmc(pos, self.nsteps,progress=True)
        # return sampler
    
    def get_chain(self,filename):
        sampler = self.sampler(filename)
        samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return samples

    def corner_plot(self,filename):
        chains = self.get_chain(filename)
        c = ChainConsumer().add_chain(chains, parameters=self.likelihood.theory.labels)
        fig = c.plotter.plot(truth=list(self.MLE(False)))
        fig.set_size_inches(3 + fig.get_size_inches()) 
        return fig


    
