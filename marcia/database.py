from numpy import loadtxt
import os


__path__ = os.path.dirname(os.path.realpath(__file__))
__datapath__ = os.path.join(__path__, 'Data')

class Data:

    def __init__(self,data:str) -> None:
        assert data in ['CC']
        self.__data_dic__ = {
                            'CC': ['CC.txt','z','cc','dcc'],
                            }
        self.data = data
        self.datafile = loadtxt(os.path.join(__datapath__, self.__data_dic__[data][0]))

    def __getattr__(self,name: str):
        if name in self.__data_dic__[self.data][1:]:
            return self.datafile[:,self.__data_dic__[self.data].index(name)-1]
        else:
            #TODO: print available attributes
            raise AttributeError(f'{name} is not a valid attribute for {self.data}') 

