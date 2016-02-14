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

    def test_features(self):
        from openpathsampling.features import coordinates as f_coordinates
        from dynamiq_engine.features import momenta as f_momenta
        sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                            p0=[1.0], alpha_p=[2.0])
        assert_equal(sampler.__features__, [f_coordinates, f_momenta])
        assert_equal(sampler.feature_dofs, {f_coordinates : None,
                                            f_momenta : None})
        sampler2 = GaussianInitialConditions(x0=[], alpha_x=[],
                                             p0=[1.0], alpha_p=[2.0],
                                             coordinate_dofs=[],
                                             momentum_dofs=[1])
        assert_equal(sampler2.__features__, [f_coordinates, f_momenta])
        assert_equal(sampler2.feature_dofs, {f_coordinates : [],
                                             f_momenta : [1]})


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

class testMMSTElectronicGaussianInitialConditions(object):
    def setup(self):
        mmst_matrix = dynq.NonadiabaticMatrix([[2.0, 1.0], [1.0, 3.0]])
        mmst = dynq.potentials.MMSTHamiltonian(mmst_matrix)
        topology = dynq.Topology(masses=np.array([]), potential=mmst)
        self.snap0x0 = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([0.0, 0.0]),
            electronic_momenta=np.array([0.0, 0.0]),
            topology=topology
        )
        self.snap0x5 = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([0.5, 0.0]),
            electronic_momenta=np.array([0.5, 0.0]),
            topology=topology
        )
        self.snap1x0 = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([1.0, 0.0]),
            electronic_momenta=np.array([1.0, 0.0]),
            topology=topology
        )
        self.snap1x5 = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([1.5, 0.0]),
            electronic_momenta=np.array([1.5, 0.0]),
            topology=topology
        )

    def test_sampler(self):
        sampler = MMSTElectronicGaussianInitialConditions.with_n_dofs(2)
        norm = 0.101321183642338
        tests = {
            self.snap0x0 : norm, 
            # 2.0 bc 2 dofs contribute
            self.snap0x5 : norm*np.exp(-2.0*(0.5**2)),
            self.snap1x0 : norm*np.exp(-2.0*(1.0**2)),
            self.snap1x5 : norm*np.exp(-2.0*(1.5**2))
        }
        check_function(sampler, tests)

        snap = sampler.generate_initial_snapshot(self.snap0x0)
        assert_not_equal(snap, self.snap0x0)
        assert_equal(snap.topology, self.snap0x0.topology)
        for d in range(2):
            assert_not_equal(snap.electronic_coordinates[d],
                             self.snap0x0.electronic_coordinates[d])
            assert_not_equal(snap.electronic_momenta[d],
                             self.snap0x0.electronic_momenta[d])

    def test_features(self):
        from openpathsampling.features import coordinates as f_coordinates
        from dynamiq_engine.features import momenta as f_momenta
        from dynamiq_engine.features import electronic_coordinates \
                as f_e_coordinates
        from dynamiq_engine.features import electronic_momenta \
                as f_e_momenta

        sampler = MMSTElectronicGaussianInitialConditions.with_n_dofs(2)
        assert_equal(sampler.__features__, [f_e_coordinates, f_e_momenta])
        assert_equal(sampler.feature_dofs, {f_e_coordinates : None,
                                            f_e_momenta : None})
        sampler2 = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                             p0=[1.0], alpha_p=[2.0])
        assert_equal(sampler.__features__, [f_e_coordinates, f_e_momenta])
        assert_equal(sampler2.__features__, [f_coordinates, f_momenta])
        
