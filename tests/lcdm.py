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
    
    


