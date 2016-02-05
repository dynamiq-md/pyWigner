import pywigner as lsc
import numpy as np
from pywigner.tools import GaussianFunction
from pywigner.tests.tools import *

class testGaussianFunction(object):
    def setup(self):
        self.gaussian = GaussianFunction(x0=[0.0, 1.0], alpha=[1.0, 2.0])

    @raises(AssertionError)
    def test_bad_setup(self):
        gaussian = GaussianFunction(x0=[1.0], alpha=[1.0, 2.0])

    def test_setup(self):
        # norm = sqrt(1.0/pi) * sqrt(2.0/pi) = sqrt(2.0) / pi 
        #      = 0.4501581580785531
        assert_almost_equal(self.gaussian.norm, 0.4501581580785531)

    def test_gaussian(self):
        tests = {
            0.0 : 0.4501581580785531*np.exp(-2.0*(-1.0)**2),
            0.5 : 0.4501581580785531*np.exp(-(0.5)**2 - 2.0*(0.5-1.0)**2),
            1.0 : 0.4501581580785531*np.exp(-(1.0)**2),
            1.5 : 0.4501581580785531*np.exp(-(1.5)**2 - 2.0*(1.5-1.0)**2)
        }
        check_function(self.gaussian, tests)


    def test_draw_samples(self):
        sample = self.gaussian.draw_sample()
        assert_equal(len(sample), 2)
        # Can't really test anything else here. See
        # visual_inspection/gaussian_function.ipynb for sampling test

