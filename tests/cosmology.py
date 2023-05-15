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
        arr = np.array([  70.03153638,  131.47733083,  229.5426088 ,  351.06529573,
                          491.02397774,  646.85808407,  816.93763436, 1000.09137079,
                          1195.41653112, 1402.18563019])
        assert np.allclose(self.theory.hubble_rate( [70, 0.3], self.zlist), arr)

    def test_transverse_distance(self):
        arr = np.array([4.27307206e+00, 3.56727608e+03, 5.48514650e+03, 6.65657529e+03,
                        7.45774195e+03, 8.04816315e+03, 8.50601569e+03, 8.87429913e+03,
                        9.17878548e+03, 9.43595655e+03])
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
        arr = np.array([4.27746823e+00, 3.70054926e+03, 5.67017552e+03, 6.85591728e+03,
                        7.66211199e+03, 8.25467114e+03, 8.71356476e+03, 9.08240786e+03,
                        9.38721924e+03, 9.64458947e+03])
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
        arr = np.array([  70.01686233,  127.89567272,  226.9968095 ,  349.25481344,
                          489.66970756,  645.80057218,  816.08389501,  999.38406754,
                          1194.81840896, 1401.67138477])
        assert np.allclose(self.theory.hubble_rate( [70, 0.3, -1.2, 0.2], self.zlist), arr)
    
    def test_transverse_distance(self):
        arr = np.array([4.27744166e+00, 3.67755622e+03, 5.63193633e+03, 6.81265266e+03,
                        7.61693360e+03, 8.20863525e+03, 8.66709615e+03, 9.03570110e+03,
                        9.34037226e+03, 9.59765511e+03])
        assert np.allclose(self.theory.transverse_distance( [70, 0.3, -1.2, 0.2], self.zlist), arr)
    
    def test_rd(self):
        assert np.allclose(round(self.theory.sound_horizon([70, 0.3, -1.2, 0.2]),2), 146.61)

