import pywigner as lsc
import dynamiq_engine as dynq
import numpy as np
from pywigner.tests.tools import *
from pywigner.samplers import *

class testGaussianInitialConditions(object):
    def setup(self):
        from dynamiq_engine.tests.stubs import PotentialStub
        topology = dynq.Topology(masses=np.array([0.5]),
                                 potential=PotentialStub())
        self.snap0x0 = dynq.Snapshot(coordinates=np.array([0.0]),
                                     momenta=np.array([0.0]),
                                     topology=topology)
        self.snap0x5 = dynq.Snapshot(coordinates=np.array([0.5]),
                                     momenta=np.array([0.5]),
                                     topology=topology)
        self.snap1x0 = dynq.Snapshot(coordinates=np.array([1.0]),
                                     momenta=np.array([1.0]),
                                     topology=topology)
        self.snap1x5 = dynq.Snapshot(coordinates=np.array([1.5]),
                                     momenta=np.array([1.5]),
                                     topology=topology)

        topol_2 = dynq.Topology(masses=np.array([0.5, 0.5]),
                                potential=PotentialStub(n_spatial=2))
        self.snap2_0x0 = dynq.Snapshot(coordinates=np.array([0.0, 0.0]),
                                       momenta=np.array([0.0, 0.0]),
                                       topology=topol_2)
        self.snap2_0x5 = dynq.Snapshot(coordinates=np.array([0.0, 0.5]),
                                       momenta=np.array([0.0, 0.5]),
                                       topology=topol_2)
        self.snap2_1x0 = dynq.Snapshot(coordinates=np.array([0.0, 1.0]),
                                       momenta=np.array([0.0, 1.0]),
                                       topology=topol_2)
        self.snap2_1x5 = dynq.Snapshot(coordinates=np.array([0.0, 1.5]),
                                       momenta=np.array([0.0, 1.5]),
                                       topology=topol_2)


    def test_sampler(self):
        # tests a sampler using all DOFs -- includes tests of all parts
        sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                            p0=[1.0], alpha_p=[2.0])
        # test that the function call is correct
        norm = 0.4501581580785531
        tests = {
            self.snap0x0 : norm*np.exp(-2.0*(-1.0)**2),
            self.snap0x5 : norm*np.exp(-(0.5)**2 - 2.0*(0.5-1.0)**2),
            self.snap1x0 : norm*np.exp(-(1.0)**2),
            self.snap1x5 : norm*np.exp(-(1.5)**2 - 2.0*(1.5-1.0)**2)
        }
        check_function(sampler, tests)

        # test that the sample generator works
        snap = sampler.generate_initial_snapshot(self.snap0x0)
        assert_not_equal(snap, self.snap0x0)
        assert_equal(snap.topology, self.snap0x0.topology)
        assert_not_equal(snap.coordinates, self.snap0x0.coordinates)
        assert_not_equal(snap.momenta, self.snap0x0.momenta)


    def test_sampler_single_coordinate(self):
        # tests a sampler using fixed momenta, partial coordinates
        sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                            p0=[], alpha_p=[],
                                            coordinate_dofs=[1],
                                            momentum_dofs=[])
        # test that the function call is correct
        norm = 0.5641895835477563
        tests = {
            self.snap2_0x0 : norm*np.exp(-1.0*(0.0)**2),
            self.snap2_0x5 : norm*np.exp(-1.0*(0.5)**2),
            self.snap2_1x0 : norm*np.exp(-1.0*(1.0)**2),
            self.snap2_1x5 : norm*np.exp(-1.0*(1.5)**2)
        }
        check_function(sampler, tests)

        # test that the sample generator works
        snap = sampler.generate_initial_snapshot(self.snap2_0x0)
        assert_equal(snap.topology, self.snap2_0x0.topology)
        assert_equal(snap.coordinates[0], self.snap2_0x0.coordinates[0])
        assert_not_equal(snap.coordinates[1], self.snap2_0x0.coordinates[1])
        assert_array_almost_equal(snap.momenta, self.snap2_0x0.momenta)


    def test_sampler_single_momentum(self):
        #tests a sampler using fixed coordinates
        sampler = GaussianInitialConditions(x0=[], alpha_x=[],
                                            p0=[1.0], alpha_p=[2.0],
                                            coordinate_dofs=[],
                                            momentum_dofs=[1])
        # test that the function call is correct
        norm = 0.7978845608028654
        tests = {
            self.snap2_0x0 : norm*np.exp(-2.0*(0.0-1.0)**2),
            self.snap2_0x5 : norm*np.exp(-2.0*(0.5-1.0)**2),
            self.snap2_1x0 : norm*np.exp(-2.0*(1.0-1.0)**2),
            self.snap2_1x5 : norm*np.exp(-2.0*(1.5-1.0)**2)
        }
        check_function(sampler, tests)

        # test that the sample generator works
        snap = sampler.generate_initial_snapshot(self.snap2_0x0)
        assert_equal(snap.topology, self.snap2_0x0.topology)
        assert_array_almost_equal(snap.coordinates, self.snap2_0x0.coordinates)
        assert_equal(snap.momenta[0], self.snap2_0x0.momenta[0])
        assert_not_equal(snap.momenta[1], self.snap2_0x0.momenta[1])

