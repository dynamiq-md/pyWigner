import openpathsampling as paths
import pywigner as lsc
import dynamiq_engine as dynq
import numpy as np
from pywigner.operators import *


from nose.tools import (
    raises, assert_equal, assert_almost_equal, assert_not_equal
)
from nose.plugins.skip import SkipTest
import dynamiq_engine.tests as dynq_tests

class OperatorTester(object):
    def __init__(self):
        pot = dynq_tests.stubs.PotentialStub(n_spatial=2)
        self.topology = dynq.Topology(masses=np.array([0.5, 2.0]), 
                                 potential=pot)
        self.previous_trajectory = [
            dynq.Snapshot(
                coordinates=np.array([0.0, 0.0]),
                momenta=np.array([0.0, 0.0]),
                topology=self.topology
            ),
            dynq.Snapshot(
                coordinates=np.array([0.1, 0.0]),
                momenta=np.array([0.1, 0.0]),
                topology=self.topology)
        ]


class testOperator(OperatorTester):
    def setup(self):
        self.op = Operator()

    @raises(NotImplementedError)
    def test_sample_initial_conditions(self):
        self.op.sample_initial_conditions(self.previous_trajectory)

    @raises(NotImplementedError)
    def test_call(self):
        self.op.sample_initial_conditions(self.previous_trajectory[0])

class testCoherentProjection(OperatorTester):
    def setup(self):
        snap0 = dynq.Snapshot(
            coordinates=np.array([1.5, 1.0]),
            momenta=np.array([0.5, 3.0]),
            topology=self.topology
        )
        self.op = CoherentProjection(
            x0=snap0.coordinates,
            p0=snap0.momenta,
            gamma=np.array([4.0, 5.0])
        )
        self.dof_op = CoherentProjection(
            x0=np.array([1.0]),
            p0=np.array([0.0]),
            gamma=np.array([1.5]),
            dofs=[1]
        )

    def test_initialization(self):
        # do we set up gamma, inv_gamma, etc correctly? (ravelled)
        for (g, i_g) in zip(self.op.gamma, self.op.inv_gamma):
            assert_almost_equal(g, 1.0/i_g)
        # do we set up n_dofs correctly? (for things with fixed dofs)
        assert_equal(self.dof_op.n_dofs, 1)
        for op in [self.op, self.dof_op]:
            assert_equal(len(op.x0), op.n_dofs)
            assert_equal(len(op.p0), op.n_dofs)
            assert_equal(len(op.gamma), op.n_dofs)
            assert_equal(len(op.inv_gamma), op.n_dofs)

    def test_set_excitons(self):
        raise SkipTest
    
    def test_sample_initial_conditions(self):
        # not much we can do here except check that the resulting snapshot
        # exists and has the right number of attributes
        #snap = self.op.sample_initial_conditions(self.previous_trajectory)
        raise SkipTest

    def test_correction(self):
        sampler = self.op.default_sampler()
        assert_equal(self.op.correction(self.previous_trajectory[0], sampler),
                     1.0)

    def test_call(self):
        raise SkipTest

    def test_excited(self):
        raise SkipTest

    def test_with_paths_snapshot(self):
        # NOTE: this will have to wait until `paths.Snapshot` has a
        # `momenta` property.
        raise SkipTest



