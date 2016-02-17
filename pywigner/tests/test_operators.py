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
        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.5, 1.0]),
            momenta=np.array([0.5, 3.0]),
            topology=self.topology
        )
        self.op = CoherentProjection(
            x0=self.snap0.coordinates.copy(),
            p0=self.snap0.momenta.copy(),
            gamma=np.array([4.0, 5.0])
        )
        self.dof_op = CoherentProjection(
            x0=np.array([1.0]),
            p0=np.array([3.0]),
            gamma=np.array([5.0]),
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
        assert_equal(self.op.excitons, [0,0])
        assert_equal(self.op._excitons, [0,0])
        assert_equal(self.op._exciton_dict, {})

        self.op.excitons = [1, 0]
        assert_equal(self.op.excitons, [1,0])
        assert_equal(self.op._excitons, [1,0])
        assert_equal(self.op._exciton_dict, {0: 1})

        self.op.excite(dof=0, excitons=0)
        assert_equal(self.op.excitons, [0,0])
        assert_equal(self.op._excitons, [0,0])
        assert_equal(self.op._exciton_dict, {})
    
    def test_sample_initial_conditions(self):
        # not much we can do here except check that the resulting snapshot
        # exists and has the right number of attributes
        sampler = self.op.default_sampler()
        snap = sampler.generate_initial_snapshot(self.snap0)

    def test_correction(self):
        sampler = self.op.default_sampler()
        assert_almost_equal(
            self.op.correction(self.previous_trajectory[0], sampler), 
            2.0**self.op.n_dofs
        )

    def test_call(self):
        norm_op = 2.0**2
        norm_dof_op = 2.0**1
        assert_almost_equal(self.op(self.snap0), norm_op)
        assert_almost_equal(self.dof_op(self.snap0), norm_dof_op)

        snap = dynq.Snapshot(
            coordinates=np.array([1.0, 0.75]),
            momenta=np.array([2.0, 6.0]),
            topology=self.topology
        )
        # dof0: exp(-4.0*(1.0-1.5)^2 - 1/4.0*(2.0-0.5)^2) = 0.209611387151098
        # dof1: exp(-5.0*(0.75-1.0)^2 - 1/5.0*(6.0-3.0)^2) = 0.120935250070417
        # dof1 * dof2 = 0.0253494055227250
        assert_almost_equal(self.op(snap), norm_op*0.0253494055227250)
        assert_almost_equal(self.dof_op(snap), norm_dof_op*0.120935250070417)
        

    def test_excited(self):
        snap = dynq.Snapshot(
            coordinates=np.array([1.0, 0.75]),
            momenta=np.array([2.0, 6.0]),
            topology=self.topology
        )
        unexcited = self.op(snap)
        self.op.excite(dof=1)
        # excited correction = 2*((0.75-1.0)**2 + (6.0-3.0)**2 -1) = 17.125
        assert_almost_equal(
            self.op._call_excited_part(snap.coordinates, snap.momenta),
            17.125
        )
        assert_almost_equal(self.op(snap) / unexcited, 17.125)

        self.op.excite(dof=1, excitons=0)
        assert_almost_equal(self.op(snap), unexcited)

    def test_with_paths_snapshot(self):
        # NOTE: this will have to wait until `paths.Snapshot` has a
        # `momenta` property.
        raise SkipTest


class testElectronicCoherentProjection(OperatorTester):
    def setup(self):
        self.op = ElectronicCoherentProjection.with_n_dofs(2).excite(1)
        self.snap0 = dynq.MMSTSnapshot(
            coordinates=np.array([]),
            momenta=np.array([]),
            electronic_coordinates=np.array([0.5, 0.75]),
            electronic_momenta=np.array([0.4, 0.3]),
            topology=None
        )

    def test_call(self):
        norm_op = 2.0**2
        # dof0 : exp(-(0.5)**2 - (0.4)**2) = 0.663650250136319
        # dof1 : exp(-(0.75)**2 - (0.3)**2) = 0.520742292353521
        # dof1_excited = 2*(0.75**2 + 0.3**2 - 0.5) = 0.305
        # standard = dof0*dof1 = 0.345590752576974
        # total = stadard*excited= 0.105405179535977
        assert_almost_equal(self.op(self.snap0), norm_op*0.105405179535977)

    def test_sample_initial_conditions(self):
        sampler = self.op.default_sampler()
        snap = sampler.generate_initial_snapshot(self.snap0)
