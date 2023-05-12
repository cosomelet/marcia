import os
import sys
import cmath
import numpy as np
import scipy as sp
from scipy.integrate import quad
import scipy.constants as const
import time

from marcia import Data


class Kernels(object):
    """
        General Gaussian process modelling that all the specific kernels must be inherited from.
        Inputs would be the appropriate amplitude and scaling parameters needed for the particular model. 

        """

    def __init__(self, model, parameters, data):

        self.clight = 299792458. / 1000.  # km/s

        self.kernels = ['SE', 'M92', 'M72']  # Possible kernel choices

        # The data must contain the list/lists of x-axes
        self.data = data
        self.ndata = len(data)
        # The possible model cases are ['SE', 'SE', 'SE'] etc. for 3 tasks and ['SE', 'SE', 'SE', 'self-scaled'] for 3 tasks and a self-scaled GP
        self.model = model
        self.nmodel = len(model)
        self.params = parameters
        self.nparams = len(parameters)

        if 'self-scaled' in self.model and (self.nparams == self.nmodel):
            self.ntasks = self.nmodel - 1
            self.model = self.model[:-1]
            # Implying that the scale length for all the datasets is set to be the same, we need sigma_f equal to the number of GPs and one l_s
            for i in range(self.ntasks-1):
                self.params = np.append(self.params, parameters[-1])

        elif 'self-scaled' not in model and (self.nparams == 2*len(self.model)):
            # We have the number of parameters matching the number of GP tasks
            self.ntasks = len(self.model)

        elif 'self-scaled' not in model and (self.nparams != 2*len(model)):
            # We need atleast two paramters per GP task: {sigma_f, l_s}
            print('Error: Number of parameters does not match the model specifications')
            sys.exit(0)

        # Reshaping the parameters to be a 2D array of shape (2, ntasks)
        self.params = np.transpose(np.reshape(np.params, (self.ntasks, 2)))

        # To create a dictionary of paramters and datasets
        self.data_f = {}
        self.sig_f = {}
        self.l_s = {}
        self.kernel_f = {}

        if len(self.data) != self.ntasks:
            print('Error: data length does not match the number of kernels')
            sys.exit(0)
        else:
            for i in range(self.ntasks):
                if self.model[i] in self.kernels and len(self.params[i]) == 2:
                    self.kernel_f['task_{0}'.format(i)] = self.model[i]
                    self.sig_f['task_{0}'.format(i)] = self.params[i, 0]
                    self.l_s['task_{0}'.format(i)] = self.params[i, 1]
                    # To create a dictionary of data
                    self.data_f['task_{0}'.format(i)] = self.data[i]

                else:
                    print(self.model[i], self.params[i])
                    print('Error: model or parameters not specified correctly')
                    sys.exit(0)

        # self.CovMat = self.Cov_Mat()

        self.CovMat = self.Cov_Mat()

    # To define the covariance matrix of the data
    def Cov_Mat(self):
        #for key_1, data_1 in self.data_f.items():
        #    for key_2, data_2 in self.data_f.items():
        raise NotImplementedError

                

    # This is to define the different possible kernel choices


    # To define the all the basis functions of all the possible kernels
    def basis_function(self, x1, x2):
        if self.model == 'SE':
            return self.s_f**2. * np.exp(- ((x1-x2)**2.)/(2. * self.l**2.))
        elif self.model == 'M92':
            return self.s_f**2. * np.exp(-np.sqrt(((x1-x2)**2.)/(2. * self.l**2.)))
        elif self.model == 'M72':
            return self.s_f**2. * np.exp(-np.sqrt(((x1-x2)**2.)/(2. * self.l**2.)))

    # # Covarinace matrix of data
    # def Inv_Cov_Mat(self):
    #     z_CC = self.data_cc.z
    #     Sigma_CC = np.diag(self.data_cc.dcc**2)
    #     cmatrix = np.reshape([0.0]*(len(z_CC)*len(z_CC)),
    #                          (len(z_CC), len(z_CC)))
    #     for i in range(len(z_CC)):
    #         for j in range(i+1):
    #             # print self.kernel(z_CC[i],z_CC[j])
    #             cmatrix[i, j] = self.kernel(z_CC[i], z_CC[j])
    #     out = cmatrix.T + cmatrix
    #     np.fill_diagonal(out, np.diag(cmatrix))
    #     covmat = np.array(out) + np.array(Sigma_CC)
    #     self.DetCovMat = np.linalg.det(covmat)
    #     return np.linalg.inv(covmat)

    # # Covariance matrix of prediction star
    # def Cov_Mat_S(self, x1):
    #     z_CC = self.data_cc.z
    #     cmat_Star = [0.0]*(len(z_CC))
    #     for i in range(len(z_CC)):
    #         cmat_Star[i] = self.kernel(x1, z_CC[i])
    #     return cmat_Star

    # # Covariance matrix of prediction star
    # def Cov_Mat_SS(self, x1):
    #     zp = [x1]
    #     # np.linspace(0.,2.5,50)
    #     cmatrix = np.reshape([0.0]*(len(zp)*len(zp)), (len(zp), len(zp)))
    #     for i in range(len(zp)):
    #         for j in range(i+1):
    #             cmatrix[i, j] = self.kernel(zp[i], zp[j])
    #     cmat_Star_Star = cmatrix.T + cmatrix
    #     np.fill_diagonal(cmat_Star_Star, np.diag(cmatrix))

    #     return cmat_Star_Star
