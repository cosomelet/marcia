import emcee
import numpy as np
from marcia import Likelihood as lk
# from marcia import Likelihood_GP as lkGP
from getdist import plots, MCSamples
import scipy.optimize as op
from chainconsumer import ChainConsumer
import logging
import os

class Sampler:

    def __init__(self, model, parameters, data, initial_guess, prior_file=None,
                 max_n=100000, nwalkers=100, sampler_file='sampler.h5', converge=False, prior_dist = 'uniform', burnin = 0.3, thin = 1):

        self.data = data
        self.model = model
        self.likelihood = lk(model, parameters, data, prior_file)
        self.ndim = len(self.likelihood.priors)
        self.nwalkers = nwalkers
        self.initial_guess = initial_guess
        self.max_n = max_n
        self.HDFBackend = emcee.backends.HDFBackend(sampler_file)
        self.converge = converge
        self.mle = {}
        self.priors = np.array(self.likelihood.priors)
        self.prior_dist = prior_dist

        # default burnin and thinning
        self.burnin = burnin
        self.thin = thin

    def MLE(self, verbose=True):
        if 'result' not in self.mle:
            nll = lambda x: -1 * self.likelihood.logProb(x)
            result = op.minimize(nll, x0=self.initial_guess, method='Nelder-Mead', options={'maxfev': None})
            if verbose:
                print(f'Best-fit values: {result.x}')
                print(f'Max-Likelihood value (including prior likelihood): {self.likelihood.logProb(result.x)}')
            self.mle['result'] = result.x

        return self.mle['result']

    def sampler_pos(self, prior_dist='uniform'):

        if prior_dist == 'uniform':
            pos = np.random.uniform(self.priors[:,0], self.priors[:,1], size = (self.nwalkers,self.ndim))
        elif prior_dist == 'best-fit':
            mle = self.MLE()
            pos = [mle + 1e-4 * np.random.randn(self.ndim) for _ in range(self.nwalkers)]
        return pos

    def sampler(self,reset=False):
        try:
            self.HDFBackend.iteration
        except OSError:
            self.HDFBackend.reset(self.nwalkers, self.ndim)

        last_iteration = self.HDFBackend.iteration if self.HDFBackend.iteration is not None else 0

        if last_iteration < self.max_n:
            if last_iteration == 0:
                print('Sampling begins')
            else:
                if reset:
                    print(f'Reseting sampling from iteration: {last_iteration}')
                    self.HDFBackend.reset(self.nwalkers, self.ndim)
                else:
                    print(f'Sampling resuming from iteration: {last_iteration}')
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood.logProb, backend=self.HDFBackend)
            if self.converge:
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
                    if converged:
                        print(f'Converged at iteration {sampler.iteration}')
                        print(f'I:{sampler.iteration}, A:{(tau*100)-sampler.iteration}, T:{np.abs(old_tau - tau) / tau}')
                        break
                    old_tau = tau
            else:
                sampler.run_mcmc(self.sampler_pos(prior_dist = self.prior_dist), self.max_n, progress=True)
        else:
            if reset:
                print(f'Reseting sampling from iteration: {last_iteration}')
                self.HDFBackend.reset(self.nwalkers, self.ndim)
                return self.sampler()
            print(f'Chain exists with {last_iteration} iterations: increase iterations to continue sampling')

    def get_burnin(self):
        try:
            tau = self.HDFBackend.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            print(f'Burn-in: {burnin} and thin: {thin}')
        except:

            print('Autocorrelation time could not be calculated, increase the number of iterations')
            burnin = self.burnin
            thin = self.thin
            print(f'Imposing default Burn-in: {burnin} and thin: {thin} by hand')
        return burnin, thin

    def get_chain(self, getdist=False):
        sampler = self.HDFBackend
        if sampler.iteration < self.max_n:
            print(f'Only {sampler.iteration} iterations completed')
            print(f'You should run the sampler to finsih the sampling of {self.max_n} iterations')
        burnin, thin = self.get_burnin()
        samples = sampler.get_chain(discard=int(burnin*self.max_n), thin=thin, flat=True)
        if getdist:
            lnprob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            lnprior = sampler.get_blobs(discard=burnin, thin=thin, flat=True)
            if lnprior is None:
                lnprior = np.zeros_like(lnprob)
            samples = np.concatenate((lnprob[:, None], lnprior[:, None], samples), axis=1)
        return samples

    def corner_plot(self, getdist=False):
        chains = self.get_chain()
        names = self.likelihood.theory.param.parameters
        labels = [p.replace('$', '') for p in self.likelihood.theory.labels]
        if getdist:
            samples = MCSamples(samples=chains, names=names, labels=labels)
            g = plots.get_subplot_plotter()
            g.triangle_plot([samples], filled=True)
        else:
            c = ChainConsumer().add_chain(chains, parameters=self.likelihood.theory.labels)
            fig = c.plotter.plot(truth=list(self.MLE(False)))
            fig.set_size_inches(3 + fig.get_size_inches())
    
    def get_simple_stat(self):
        samples = self.get_chain()
        names = self.likelihood.theory.param.parameters
        data = {}
        for i,name in enumerate(names):
            data[name] = {}
            data[name]['mean'] = np.mean(samples[:,i])
            data[name]['std'] = np.std(samples[:,i])
        return data


# class GPSampler(Sampler):
#     """
#     Class to sample the GP parameters that inherits from the Sampler class
#     """
#     def __init__(self, GP_filename = None):
#         super().__init__()
#         data_list = self.data

#         if GP_filename is None:
#             GP_filename = 'GPconfig.ini'

#         self.likelihood = lkGP(data = data_list, GP_filename = GP_filename)