import sys
import os 
import numpy as np
real_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{real_path}/../')
from marcia import Cosmology
import unittest

class TestLCDM(unittest.TestCase):
    def setUp(self):
        self.theory = Cosmology('LCDM',['H0','Omega_m'])
        self.z_min = 0.001
        self.z_max = 10.0
        self.zlist = np.linspace(self.z_min,self.z_max,10)

    def test_hubble_rate(self):
        arr = np.array([ 70.03153638,  131.47733083,  229.5426088 ,  351.06529573,
                        491.02397774,  646.85808407,  816.93763436, 1000.09137079,
                        1195.41653112, 1402.18563019])
        assert np.allclose(self.theory.hubble_rate( [70, 0.3], self.zlist), arr)

    def test_transverse_distance(self):
        arr = np.array([   0.        , 3563.00252552, 5480.86846219, 6652.2955333 ,
                        7453.46122345, 8043.88164985, 8501.73372259, 8870.0167495 ,
                         9174.50291195, 9431.67380384])
        assert np.allclose(self.theory.transverse_distance( [70, 0.3], self.zlist), arr)
    
    def test_rd(self):
        assert np.allclose(round(self.theory.sound_horizon([70, 0.3]),2), 146.61)
    
    
class TestwCDM(unittest.TestCase):
    
    def setUp(self):
        self.theory = Cosmology('wCDM',['H0','Omega_m','w0'])
        self.z_min = 0.001
        self.z_max = 10.0
        self.zlist = np.linspace(self.z_min,self.z_max,10)

    def test_hubble_rate(self):
        arr = np.array([  70.01685499,  126.67521252,  225.74229982,  348.19526433,
                          488.78992978,  645.0625029 ,  815.45561061,  998.84168575,
                          1194.34432306, 1401.25249321])
        assert np.allclose(self.theory.hubble_rate( [70, 0.3, -1.2], self.zlist), arr)

    def test_transverse_distance(self):
        arr = np.array([   0.        , 3696.26106198, 5665.87144101, 6851.58369802,
                           7657.7741683 , 8250.33207584, 8709.22521719, 9078.06811247,
                           9382.87912752, 9640.24985304])
        assert np.allclose(self.theory.transverse_distance( [70, 0.3, -1.2], self.zlist), arr)
    
    def test_rd(self):
        assert np.allclose(round(self.theory.sound_horizon([70, 0.3, -1.2]),2), 146.61)

class TestCPL(unittest.TestCase):

    def setUp(self):
        self.theory = Cosmology('CPL',['H0','Omega_m','w0','wa'])
        self.z_min = 0.001
        self.z_max = 10.0
        self.zlist = np.linspace(self.z_min,self.z_max,10)
    
    def test_hubble_rate(self):
        arr = np.array([   70.01686233,  127.89567272,  226.9968095 ,  349.25481344,
                           489.66970756,  645.80057218,  816.08389501,  999.38406754,
                           1194.81840896, 1401.67138477])
        assert np.allclose(self.theory.hubble_rate( [70, 0.3, -1.2, 0.2], self.zlist), arr)
    
    def test_transverse_distance(self):
        arr = np.array([   0.        , 3673.28337592, 5627.65857786, 6808.37226025,
                           7612.65180016, 8204.35264164, 8662.81301718, 9031.41762285,
                           9336.08855195, 9593.37122972])
        assert np.allclose(self.theory.transverse_distance( [70, 0.3, -1.2, 0.2], self.zlist), arr)
    
    def test_rd(self):
        assert np.allclose(round(self.theory.sound_horizon([70, 0.3, -1.2, 0.2]),2), 146.61)

