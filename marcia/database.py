from numpy import loadtxt
import numpy as np
import os


__path__ = os.path.dirname(os.path.realpath(__file__))
__datapath__ = os.path.join(__path__,'../' 'Data')

class Data:

    def __init__(self,data):
        datalist = ['CC','BAO']
        if type(data) is str:
            assert data in datalist, f'{data} is not in {datalist}'
            self.data = [data]
        elif type(data) is list:
            raise NotImplementedError
            # for d in data:
            #     assert d in datalist, f'{d} is not in {datalist}'
            # self.data = data
        else:
            raise TypeError(f'{data} is not a valid type')
        
        self.x,self.y,self.covar = self.get_data()

    def get_data(self):
        for d in self.data:
            if d == 'CC':
                return self.get_cosmic_clocks()
            elif d == 'BAO':
                return self.get_bao()
            elif d == 'GR':
                return self.get_growth()
            elif d == 'Lya':
                return self.get_Lya()
    
    def get_cosmic_clocks(self):
        datafile = loadtxt(os.path.join(__datapath__, 'Cosmic_Clocks','CC.txt'))
        x = datafile[:,0]
        y = datafile[:,1]
        sigma = datafile[:,2]
        covar = np.diag(sigma**2)
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar
    
    def get_bao(self):
        datafile = loadtxt(os.path.join(__datapath__, 'Alam2016','DmH.txt'))
        datafile2 = loadtxt(os.path.join(__datapath__, 'Alam2016','CovDmh.txt'))
        x = datafile[:,0]
        y = datafile[:,1]
        covar = datafile2
        assert len(x) == len(y) == covar.shape[0] == covar.shape[1]
        return x,y,covar
    
    def get_growth(self,file=0):
        datafile = loadtxt(os.path.join(__datapath__, 'Growth Rate',f'GR{file}.txt' if file > 0 else 'GR.txt')) 
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
    


