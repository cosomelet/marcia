import emcee
import numpy as np
from marcia import Likelihood as lk
from marcia import temporarily_false
from getdist import plots, MCSamples
import scipy.optimize as op
from chainconsumer import ChainConsumer
import logging
import os

logging.basicConfig(filename="sampler.log",format='%(asctime)s %(message)s',filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)



class Sampler:

    def __init__(self,model,parameters,data,initial_guess,prior_file=None,
                 max_n=100000,nwalkers=100,sampler_file='sampler.h5',
                 resume=False,converge=False):
        
        self.likelihood = lk(model,parameters,data,prior_file)
        self.ndim = len(self.likelihood.priors)
        self.nwalkers = nwalkers
        self.initial_guess = initial_guess
        self.max_n = max_n
        self.sampler_file = sampler_file
        self.resume = resume
        self.converge = converge
        self.mle = {}

    
    def MLE(self,verbose=True):
        if 'result' not in self.mle.keys():
            nll = lambda x: -1*self.likelihood.logProb(x)
            result = op.minimize(nll, x0=self.initial_guess, method = 'Nelder-Mead', options={'maxfev': None})
            if verbose:
                print(f'Best-fit values: {result.x}')
                print(f'Max-Likelihood value (including prior likelihood):{self.likelihood.logProb(result.x)}')
            self.mle['result'] = result.x
        
        return self.mle['result']
    

    def sampler_pos(self):
        mle = self.MLE()
        pos = [mle+ 1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        return pos
    
    def sampler_w_covergence(self):
        backend = emcee.backends.HDFBackend(self.sampler_file)
        if os.path.isfile(self.sampler_file) and (not self.resume):
            return backend
        else:
            if self.resume:
                print('Resuming from previous run')
            else:
                backend.reset(self.nwalkers, self.ndim)
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood.logProb, backend=backend)
            index = 0
            autocorr = np.empty(self.max_n)
            old_tau = np.inf
            for sample in sampler.sample(self.sampler_pos(), iterations=self.max_n, progress=True):

                if sampler.iteration % 100:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                logger.info(f'I:{sampler.iteration}, A:{(tau*100)-sampler.iteration}, T:{np.abs(old_tau - tau) / tau}')
                if converged:
                    print(f'Converged at iteration {sampler.iteration}')
                    break
                old_tau = tau
            
            return sampler
    def sampler_wo_covergence(self):
        backend = emcee.backends.HDFBackend(self.sampler_file)
        if os.path.isfile(self.sampler_file) and (not self.resume):
            return backend
        else:
            if self.resume:
                print('Resuming from previous run')
            else:
                backend.reset(self.nwalkers, self.ndim)
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood.logProb, backend=backend)
            sampler.run_mcmc(self.sampler_pos(), self.max_n, progress=True)
            return sampler
    
    def sampler(self):
        if self.converge:
            return self.sampler_w_covergence()
        else:
            return self.sampler_wo_covergence()

    #@temporarily_false('resume')
    def get_burnin(self):
        try:
            tau = self.sampler().get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
        except:
            burnin = 50
            thin = 1
        return burnin, thin

    @temporarily_false('resume')
    def get_chain(self,getdist=False):
        sampler = self.sampler()
        burnin, thin = self.get_burnin()
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        if getdist:
            lnprob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            lnprior = sampler.get_blobs(discard=burnin, thin=thin, flat=True)
            if lnprior is None:
                lnprior = np.zeros_like(lnprob)
            samples = np.concatenate((lnprob[:,None],lnprior[:,None],samples),axis=1)
        return samples
    
    @temporarily_false('resume')
    def corner_plot(self,getdist=False):
        chains = self.get_chain()
        names = self.likelihood.theory.param.parameters
        labels = [p.replace('$','') for p in self.likelihood.theory.labels]
        if getdist:
            samples = MCSamples(samples=chains,names=names,labels=labels)
            g = plots.get_subplot_plotter()
            g.triangle_plot([samples], filled=True)
        else:
            c = ChainConsumer().add_chain(chains, parameters=self.likelihood.theory.labels)
            fig = c.plotter.plot(truth=list(self.MLE(False)))
            fig.set_size_inches(3 + fig.get_size_inches()) 
        
    


    
