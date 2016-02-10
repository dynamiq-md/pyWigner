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
            alpha_x=[1.0, 1.0], alpha_p=[1.0, 1.0]
        )
        self.e_sampler = MMSTElectronicGaussianInitialConditions.with_n_dofs(2)
        self.sampler = OrthogonalInitialConditions([self.normal_sampler,
                                                    self.e_sampler])
        # TODO set up snapshots

    def test_features(self):
        from openpathsampling.features import coordinates as f_coordinates
        from dynamiq_engine.features import momenta as f_momenta
        from dynamiq_engine.features import electronic_coordinates \
                as f_e_coordinates
        from dynamiq_engine.features import electronic_momenta \
                as f_e_momenta
        
        assert_equal(set(self.sampler.__features__), 
                     set([f_coordinates, f_momenta, f_e_coordinates,
                          f_e_momenta]))
        pass

    def test_sampler(self):
        pass
