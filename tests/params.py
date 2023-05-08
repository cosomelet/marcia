import sys
import os 
import numpy as np
real_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{real_path}/../')
from marcia import Params
import unittest

class TestPARAM(unittest.TestCase):
    def setUp(self):
        self.param = Params(['H0'])
    
    def test_filechange(self):
        assert len(self.param.act_params['Label'].keys()) == 14

    
    def test_object(self):
        p = self.param([70])
        assert hasattr(p,'H0')
    
