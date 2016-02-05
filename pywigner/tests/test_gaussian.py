import pywigner as lsc
from pywigner.tools import GaussianFunction
from pywigner.tests.tools import *

class testGaussianFunction(object):
    def setup(self):
        self.gaussian = GaussianFunction(x0=[0.0, 1.0], alpha=[1.0, 2.0])

    @raises(AssertionError)
    def test_bad_setup(self):
        gaussian = GaussianFunction(x0=[1.0], alpha=[1.0, 2.0])

    def test_gaussian(self):
        raise SkipTest

    def test_draw_samples(self):
        raise SkipTest

    def test_set_array_to_sample(self):
        raise SkipTest
