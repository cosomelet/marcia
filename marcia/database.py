from typing import Any
from numpy import loadtxt
import numpy as np
import os
from marcia import load_data_once


__path__ = os.path.dirname(os.path.realpath(__file__))
__datapath__ = os.path.join(__path__, '..','Data')



class Data:

    def __init__(self,data,file_fs8=0,Lambda=1,b=1,sigma_sys=0.7571,H0=1,):
        datalist = ['CC','BAO-alam','BAO-zhao','GR','Lya','GRB','SNE','QSO','QSO_full', 'Planck_TT', 'Planck_EE', 'Planck_TE','Pantheon_plus','Pantheon_old']
        if type(data) is str:
            assert data in datalist, f'{data} is not in {datalist}'
            self.data = [data]
        elif type(data) is list:
            for d in data:
                assert d in datalist, f'{d} is not in {datalist}'
            self.data = data
        else:
            raise TypeError(f'{data} is not a valid type')
        
        self.file_fs8 = file_fs8
        self.Lambda = Lambda
        self.b = b
        self.sigma_sys = sigma_sys
        self.H0 = H0

        self.x = {}
        self.y = {}
        self.covar = {}

    def __block_matrix__(self,matrices):
        size = sum(matrix.shape[0] for matrix in matrices)
        result = np.zeros((size, size))

        # Fill the resulting matrix with the input matrices in a block diagonal manner
        row_start = 0
        col_start = 0
        for matrix in matrices:
            row_end = row_start + matrix.shape[0]
            col_end = col_start + matrix.shape[1]
            result[row_start:row_end, col_start:col_end] = matrix
            row_start = row_end
            col_start = col_end
        
        return result

    def __call__(self, paramdict=None):
        if paramdict is None:
            print('Nuisance parameters are set to default values')
        elif type(paramdict) is dict:
            for attr, value in paramdict.items():
                if hasattr(self, attr):
                    setattr(self, attr, value)
                else:
                    raise ValueError(f'{attr} is not a valid parameter')
        else:
            raise TypeError(f'{paramdict} is not a valid type')

        # To create dictionaries for each data and the covariance matrix and finally the joint data
        
        covar_all = []

        for i, d in enumerate(self.data):
            self.x[f'x{i}'], self.y[f'y{i}'], self.covar[f'covar{i}'] = self.get_data(d)
            covar_all.append(self.covar[f'covar{i}'])

        x_all = np.concatenate(list(self.x.values()))
        y_all = np.concatenate(list(self.y.values()))
        covar_all = self.__block_matrix__(covar_all)
        return x_all,y_all,covar_all    

    def get_data(self,choose):
        if choose == 'CC':
            return self.get_cosmic_clocks()
        elif choose == 'BAO-alam':
            return self.get_BAO_alam()
        elif choose == 'BAO-zhao':
            return self.get_BAO_zhao()
        elif choose == 'GR':
            return self.get_growth()
        elif choose == 'Lya':
            return self.get_Lya()
        elif choose == 'GRB':
            return self.get_GRB()
        elif choose == 'SNE':
            return self.get_SNE()
        elif choose == 'QSO':
            return self.get_QSO()
        elif choose == 'QSO_full':
            return self.get_QSO_full()
        elif choose == 'Planck_TT':
            return self.get_CMB_planck_TT()
        elif choose == 'Planck_EE':
            return self.get_CMB_planck_EE()
        elif choose == 'Planck_TE':
            return self.get_CMB_planck_TE()
        elif choose == 'Pantheon_plus':
            return self.get_pantheon_plus()
        else:
            raise ValueError(f'{choose} is not a valid data set')
    
    @load_data_once
    def get_data_cosmic_clocks(self):
        return loadtxt(os.path.join(__datapath__, 'Cosmic_Clocks','CC.txt'))
    
    def get_cosmic_clocks(self):
        datafile = self.get_data_cosmic_clocks()
        x = datafile[:,0]
        y = datafile[:,1]/self.H0
        sigma = datafile[:,2]
        covar = np.diag(sigma**2)
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar

    @load_data_once
    def get_BAO_alam(self, ):
        datafile = loadtxt(os.path.join(__datapath__, 'Alam2016','DmH.txt'))
        datafile2 = loadtxt(os.path.join(__datapath__, 'Alam2016','CovDmh.txt'))
        #indices = datafile[:, 0].argsort()
        x = datafile[:,0]#[indices]
        y = datafile[:,1]#[indices]
        covar = datafile2#[:, indices][indices, :]
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar
    
    @load_data_once
    def get_BAO_zhao(self,):
        datafile = loadtxt(os.path.join(__datapath__, 'Zhao2018','All.txt'))
        datafile2 = loadtxt(os.path.join(__datapath__, 'Zhao2018','Cov_All.txt'))
        x = datafile[:,0]
        y = datafile[:,1]
        covar = datafile2
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar
    
    def get_growth(self):
        datafile = loadtxt(os.path.join(__datapath__, 'Growth Rate',f'GR{self.file_fs8}.txt' if self.file_fs8 > 0 else 'GR.txt')) 
        x = datafile[:,0]
        y = datafile[:,1]
        covar = np.diag(datafile[:,2]**2)
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar
    
    def get_Lya(self):
        datafile = loadtxt(os.path.join(__datapath__, 'Lyman-alpha','DmH.txt'))
        datafile2 = loadtxt(os.path.join(__datapath__, 'Lyman-alpha','CovDmh.txt')) 
        x = datafile[:,0]
        y = datafile[:,1]
        covar = datafile2
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar
    
    def get_GRB(self):
        datafile = loadtxt(os.path.join(__datapath__, 'GRB','GRB.txt'),usecols=(1,2,3,4,5))
        z = datafile[:,0]
        S_b = datafile[:,1]
        sigma_S_b = datafile[:,2]
        E_p = datafile[:,3]
        sigma_E_p = datafile[:,4]

        mu1 = (1+z)/(4*np.pi)
        mu2 = (E_p/300)**self.b
        mu3 = (S_b**-1/100)
        mu = (5/2)*(np.log10(mu1*mu2*mu3)+ self.Lambda)

        sigma_mu1 = (5/(2*np.log(10)))**2
        sigma_mu2 = (self.b*sigma_E_p/E_p)**2
        sigma_mu3 = (sigma_S_b/S_b)**2
        sigma_mu = sigma_mu1*(sigma_mu2+sigma_mu3+self.sigma_sys**2)

        covar = np.diag(sigma_mu)

        x = z
        y = mu
        return x,y,covar

    def get_SNE(self):
        data = loadtxt(os.path.join(__datapath__, 'Pantheon_E','SNE.txt'))
        corr = loadtxt(os.path.join(__datapath__, 'Pantheon_E','SSNE.txt'))
        x = data[:,0]
        y = 1/data[:,1]
        sigma = data[:,2]
        covar = sigma * corr * sigma.T
        return x,y,covar
    
    

    
    @load_data_once
    def get_QSO_data(self):
        data = loadtxt(os.path.join(__datapath__, 'Quasars','table3.dat'), usecols=(3,4,5,6,7,9,10, 11, 12))
        position = loadtxt(os.path.join(__datapath__, 'Quasars','table3.dat'), usecols=(1,2))
        ra = position[:,0]
        dec = position[:,1]
        z, lnFUV, lnFUV_err, lnFX, lnFX_err = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
        DM, dDM = data[:,7], data[:,8]
        return z, lnFUV, lnFUV_err, lnFX, lnFX_err, DM, dDM, ra, dec
    
    def get_QSO_full(self):
        z, lnFUV, lnFUV_err, lnFX, lnFX_err, _, _ ,_,_= self.get_QSO_data()
        return z, lnFUV, lnFUV_err, lnFX, lnFX_err

    def get_QSO(self):
        x,_,_,_,_,y,sigma,_,_ = self.get_QSO_data()
        covar = np.diag(sigma**2)
        return x,y,covar
    
    def get_QSO_position(self):
        _,_,_,_,_,_,_,ra,dec = self.get_QSO_data()
        return ra,dec

    def get_CMB_planck_TT(self):
        # This is the Planck 2018 data special case where we want 
        # to use the TT data only
        data = loadtxt(os.path.join(__datapath__, 'Planck_Spectra','COM_PowerSpect_CMB-TT-binned_R3.01.txt'))
        x = data[:,0]/1000
        y = data[:,1]/1000
        sigma = (data[:,2] + data[:,3])/2/1000
        covar = np.diag(sigma**2)
        return x,y,covar
    
    def get_CMB_planck_EE(self):
        # This is the Planck 2018 data special case where we want 
        # to use the EE data only
        data = loadtxt(os.path.join(__datapath__, 'Planck_Spectra','COM_PowerSpect_CMB-EE-binned_R3.02.txt'))
        x = data[:,0]/1000
        y = data[:,1]/10
        sigma = (data[:,2] + data[:,3])/2/10
        covar = np.diag(sigma**2)
        return x,y,covar
    
    def get_CMB_planck_TE(self):
        # This is the Planck 2018 data special case where we want 
        # to use the TE data only
        data = loadtxt(os.path.join(__datapath__, 'Planck_Spectra','COM_PowerSpect_CMB-TE-binned_R3.02.txt'))
        x = data[:,0]/1000
        y = data[:,1]/100
        sigma = (data[:,2] + data[:,3])/2/100
        covar = np.diag(sigma**2)
        return x,y,covar
    
    @load_data_once
    def get_data_pantheon_old(self):  
        datapath = os.path.join(__datapath__, 'Pantheon_old','lcparam_full.txt')
        covpath = os.path.join(__datapath__, 'Pantheon_old','cov_full.txt')
        zcmb, zhel, mb, ra, dec = np.loadtxt(datapath).T
        cov = np.loadtxt(covpath)
        return zcmb, zhel, mb, cov, ra, dec
    
    def get_pantheon_old(self,Zhel=False,position=False):
        zcmb, zhel, mb, cov, ra, dec = self.get_data_pantheon_old()
        x = zcmb
        y = mb
        covar = cov
        if Zhel:
            return zhel
        elif position:
            return ra, dec
        else:
            return x,y,covar
    
    @load_data_once
    def get_data_pantheon(self):
        datapath = os.path.join(__datapath__, 'Pantheon+','Pantheon+SH0ES.dat')
        covpath = os.path.join(__datapath__, 'Pantheon+','Pantheon+SH0ES_STAT+SYS.cov')
        
        data = np.genfromtxt(datapath, skip_header=1)

   
        with open(datapath, 'r') as file:
            titles = file.readline().split() 
            index_ra = titles.index('RA')
            index_dec = titles.index('DEC')

        wh = (data[:, 2] > 0.01) # redshift cut
        zhel = data[wh, 6] # heliocentric redshift
        zcmb = data[wh, 2] # peculiar velocity corrected cmb redshift 
        mbs = data[wh, 8] # corrected appearent magnitude
        ra = data[wh, index_ra] # right ascension
        dec = data[wh, index_dec] # declination

        with open(covpath, 'r') as file:
            cov = np.loadtxt(file, skiprows=1)

        n = int(round(np.sqrt(len(cov))))
        cov = np.reshape(cov, (n, n))

        mask = np.array(wh, bool)
        cov = cov[mask][:, mask]
        return zcmb, zhel, mbs, cov, ra, dec
    
    
    def get_pantheon_plus(self,Zhel=False,position=False):
        zcmb, zhel, mbs, cov, ra, dec = self.get_data_pantheon()
        x = zcmb
        y = mbs
        covar = cov

        if Zhel:
            return zhel
        elif position:
            return ra, dec
        else:
            return x,y,covar
       
        