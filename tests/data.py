import sys
import os 
import numpy as np
real_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{real_path}/../')
from marcia import Data
import unittest

class TestDATA(unittest.TestCase):
    def setUp(self):
        self.data_cc = Data('CC')
        self.data_bao = Data('BAO')
        self.data_cc_bao = Data(['CC','BAO'])

    def test_covariance(self):
        _,_,cov_cc = self.data_cc({})
        _,_,cov_bao = self.data_bao({})
        _,_,cov_cc_bao = self.data_cc_bao({})

        assert cov_cc_bao.shape[0] == cov_cc.shape[0] + cov_bao.shape[0]
        assert cov_cc_bao.shape[1] == cov_cc.shape[1] + cov_bao.shape[1]
    
    