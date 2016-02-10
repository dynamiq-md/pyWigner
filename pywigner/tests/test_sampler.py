import pywigner as lsc
import dynamiq_engine as dynq
import numpy as np
from pywigner.tests.tools import *
from pywigner.samplers import *

class testOrthogonalInitialConditions(object):
    def setup(self):
        from dynamiq_engine.tests.stubs import PotentialStub
        topology = dynq.Topology(masses=np.array([0.5, 0.5]),
                                 potential=PotentialStub(2))
        self.normal_sampler = GaussianInitialConditions(
            x0=[0.0, 0.0], p0=[0.0, 0.0], 
            alpha_x=[2.0, 1.0], alpha_p=[2.0, 1.0]
        )
        self.e_sampler = MMSTElectronicGaussianInitialConditions.with_n_dofs(2)
        self.sampler = OrthogonalInitialConditions([self.normal_sampler,
                                                    self.e_sampler])
        # TODO set up snapshots
        self.snap0x0 = dynq.MMSTSnapshot(
            coordinates=np.array([0.0, 0.0]),
            momenta=np.array([0.0, 0.0]),
            electronic_coordinates=np.array([0.0, 0.0]),
            electronic_momenta=np.array([0.0, 0.0]),
            topology=topology
        )
        self.snap0x5 = dynq.MMSTSnapshot(
            coordinates=np.array([0.5, 0.0]),
            momenta=np.array([0.5, 0.0]),
            electronic_coordinates=np.array([0.5, 0.0]),
            electronic_momenta=np.array([0.5, 0.0]),
            topology=topology
        )
        self.snap1x0 = dynq.MMSTSnapshot(
            coordinates=np.array([1.0, 0.0]),
            momenta=np.array([1.0, 0.0]),
            electronic_coordinates=np.array([1.0, 0.0]),
            electronic_momenta=np.array([1.0, 0.0]),
            topology=topology
        )

    @raises(RuntimeError)
    def test_error_with_none_dofs_overlap(self):
        part_sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                                 p0=[], alpha_p=[],
                                                 coordinate_dofs=[1],
                                                 momentum_dofs=[])
        sampler = OrthogonalInitialConditions([self.normal_sampler,
                                               self.e_sampler,
                                               part_sampler])


    @raises(RuntimeError)
    def test_error_with_dofs_overlap(self):
        part_sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                                 p0=[], alpha_p=[],
                                                 coordinate_dofs=[1],
                                                 momentum_dofs=[])
        part_sampler2 = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                                  p0=[], alpha_p=[],
                                                  coordinate_dofs=[1],
                                                  momentum_dofs=[])
        sampler = OrthogonalInitialConditions([part_sampler, part_sampler2])


    def test_features(self):
        from openpathsampling.features import coordinates as f_coordinates
        from dynamiq_engine.features import momenta as f_momenta
        from dynamiq_engine.features import electronic_coordinates \
                as f_e_coordinates
        from dynamiq_engine.features import electronic_momenta \
                as f_e_momenta
        
        assert_equal(
            set(self.sampler.__features__), 
            set([f_coordinates, f_momenta, f_e_coordinates, f_e_momenta])
        )
        assert_equal(len(self.sampler.feature_dofs.keys()), 4)
        assert_equal(self.sampler.feature_dofs,
                     {f_coordinates : None, f_momenta : None,
                      f_e_coordinates : None, f_e_momenta : None})

        subsampler1 = MMSTElectronicGaussianInitialConditions(
            x0=[0.0], alpha_x=[1.0], p0=[], alpha_p=[],
            coordinate_dofs=[1], momentum_dofs=[]
        )
        subsampler2 = GaussianInitialConditions(
            x0=[0.0], alpha_x=[1.0], p0=[], alpha_p=[],
            coordinate_dofs=[1], momentum_dofs=[]
        )
        new_sampler = OrthogonalInitialConditions([subsampler1, subsampler2])
        assert_equal(
            set(new_sampler.__features__), 
            set([f_coordinates, f_momenta, f_e_coordinates, f_e_momenta])
        )
        assert_equal(new_sampler.feature_dofs,
                     {f_coordinates : [1], f_momenta : [],
                      f_e_coordinates : [1], f_e_momenta : []})


    def test_sampler(self):
        norm = 0.0205319645093687 # 2.0 / pi^4
        tests = {
            self.snap0x0 : norm,
            self.snap0x5 : norm*np.exp(-2.0*(0.5**2)-4.0*(0.5**2)),
            self.snap1x0 : norm*np.exp(-2.0*(1.0**2)-4.0*(1.0**2)),
        }
        check_function(self.sampler, tests)

        snap = self.sampler.generate_initial_snapshot(self.snap0x0)
        assert_not_equal(snap, self.snap0x0)
        assert_equal(snap.topology, self.snap0x0.topology)
        x_a = list(snap.coordinates.ravel())
        x_b = list(self.snap0x0.coordinates.ravel())
        p_a = list(snap.momenta.ravel())
        p_b = list(self.snap0x0.momenta.ravel())
        x_e_a = list(snap.electronic_coordinates.ravel())
        x_e_b = list(self.snap0x0.electronic_coordinates.ravel())
        p_e_a = list(snap.electronic_momenta.ravel())
        p_e_b = list(self.snap0x0.electronic_momenta.ravel())
        all_a = x_a + p_a + x_e_a + p_e_a
        all_b = x_b + p_b + x_e_b + p_e_b
        for (a, b) in zip(all_a, all_b):
            assert_not_equal(a, b)
