import pywigner as lsc
import dynamiq_engine as dynq
import numpy as np
from pywigner.tests.tools import *
from pywigner.samplers import *

class testGaussianInitialConditions(object):
    def setup(self):
        ho = dynq.potentials.interactions.HarmonicOscillatorInteraction(
            k=2.0, x0=1.0
        )
        self.potential = dynq.potentials.OneDimensionalInteractionModel(ho)
        self.topology = dynq.Topology(masses=np.array([0.5]), 
                                      potential=self.potential)

    def test_sampler(self):
        # tests a sampler using all DOFs -- includes tests of all parts
        sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                            p0=[1.0], alpha_p=[2.0])
        # test that the function call is correct
        snap0x0 = dynq.Snapshot(coordinates=np.array([0.0]), 
                                momenta=np.array([0.0]), 
                                topology=self.topology)
        snap0x5 = dynq.Snapshot(coordinates=np.array([0.5]), 
                                momenta=np.array([0.5]), 
                                topology=self.topology)
        snap1x0 = dynq.Snapshot(coordinates=np.array([1.0]), 
                                momenta=np.array([1.0]), 
                                topology=self.topology)
        snap1x5 = dynq.Snapshot(coordinates=np.array([1.5]), 
                                momenta=np.array([1.5]), 
                                topology=self.topology)
        tests = {
            snap0x0 : 0.4501581580785531*np.exp(-2.0*(-1.0)**2),
            snap0x5 : 0.4501581580785531*np.exp(-(0.5)**2 - 2.0*(0.5-1.0)**2),
            snap1x0 : 0.4501581580785531*np.exp(-(1.0)**2),
            snap1x5 : 0.4501581580785531*np.exp(-(1.5)**2 - 2.0*(1.5-1.0)**2)
        }
        check_function(sampler, tests)

        # test that the sample generator works
        snap = sampler.generate_initial_snapshot(snap0x0)
        assert_not_equal(snap, snap0x0)
        assert_equal(snap.topology, snap0x0.topology)
        assert_not_equal(snap.coordinates, snap0x0.coordinates)
        assert_not_equal(snap.momenta, snap0x0.momenta)


    def test_sampler_subdim(self):
        # tests a sampler using fixed momenta
        sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                            p0=[1.0], alpha_p=[2.0],
                                            coordinate_dofs=[0],
                                            momentum_dofs=[])
        # test that the function call is correct
        # test that the sample generator works

        #tests a sampler using fixed coordinates
        sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                            p0=[1.0], alpha_p=[2.0],
                                            coordinate_dofs=[],
                                            momentum_dofs=[0])
        pass

