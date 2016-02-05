import pywigner as lsc
import numpy as np
# dodge circular import issues
from pywigner.samplers import InitialConditionSampler


class GaussianInitialConditions(InitialConditionSampler):
    def __init__(self, x0, p0, alpha_x, alpha_p, coordinate_dofs=None,
                 momentum_dofs=None):
        self.x0 = x0
        self.p0 = p0
        self.alpha_x = alpha_x
        self.alpha_p = alpha_p
        self.coordinate_dofs = coordinate_dofs
        self.momentum_dofs = momentum_dofs

        self.coordinate_gaussian = lsc.tools.GaussianFunction(x0, alpha_x)
        self.momentum_gaussian = lsc.tools.GaussianFunction(p0, alpha_p)


    def generate_initial_snapshot(self, previous_snapshot):
        snapshot = previous_snapshot.copy() 
        # this might be a shallow copy, so deepen over the features:
        snapshot.momenta = snapshot.momenta.copy()
        snapshot.coordinates = snapshot.coordinates.copy()
        self.fill_initial_snapshot(snapshot, previous_snapshot)
        return snapshot


    def fill_initial_snapshot(self, snapshot, previous_snapshot):
        # This is separated from the `generate_initial_snapshot` function
        # because this can be reused by other functions, e.g., so that only
        # part of the total snapshot if filled by the Gaussian sampling (if
        # we want to fix one variable, for example)
        #
        # TODO: this can probably be sped up, but also probably isn't a
        # bottleneck in the overall calculation
        x0 = self.coordinate_gaussian.draw_sample()
        p0 = self.momenta_gaussian.draw_sample()
        if self.coordinate_dofs is None:
            np.copyto(snapshot.coordinates, x0)
        else:
            for (d, x) in zip(self.coordinate_dofs, x0):
                snapshot.coordinates[d] = x
        if self.momentum_dofs is None:
            np.copyto(snapshot.momentum, p0)
        else:
            for (d, p) in zip(self.coordinate_dofs, p0):
                snapshot.momenta[d] = x


    def __call__(self, snapshot):
        return (self.coordinate_gaussian(snapshot.coordinates)
                * self.momentum_gaussian(snapshot.momenta))


class MMSTElectronicGaussianInitialConditions(GaussianInitialConditions):
    @classmethod
    def with_n_dofs(cls, n_dofs):
        return MMSTElectronicGaussianSampler(
            x0=np.array([0.0]*n_dofs), p0=np.array([0.0]*n_dofs), 
            alpha_x=np.array([1.0]*n_dofs), alpha_p=np.array([1.0]*n_dofs)
        )

    def generate_trial_trajectory(self, previous_trajectory):
        pass

    def __call__(self, snapshot):
        return (self.coordinate_gaussian(snapshot.electronic_coordinates)
                * self.momentum_gaussian(snapshot.electronic_momenta))

