import pywigner as lsc
import numpy as np
# dodge circular import issues
from pywigner.samplers import InitialConditionSampler

# both of these are for the features
import openpathsampling as paths
import dynamiq_engine.features as features


class GaussianInitialConditions(InitialConditionSampler):
    __features__ = [paths.features.coordinates, features.momenta]
    def __init__(self, x0, p0, alpha_x, alpha_p, coordinate_dofs=None,
                 momentum_dofs=None):
        self.x0 = x0
        self.p0 = p0
        self.alpha_x = alpha_x
        self.alpha_p = alpha_p
        self.coordinate_dofs = coordinate_dofs
        self.momentum_dofs = momentum_dofs

        # the following also works for any extended momentum samplers; more
        # complicated combinations should be done without inheritance
        self.feature_dofs = {
            self.__features__[0] : coordinate_dofs,
            self.__features__[1] : momentum_dofs
        }

        self.coordinate_gaussian = lsc.tools.GaussianFunction(x0, alpha_x)
        self.momentum_gaussian = lsc.tools.GaussianFunction(p0, alpha_p)

    @staticmethod
    def _fill_feature(snapshot_array, sampler, dofs):
        # TODO: this can probably be sped up, but also probably isn't a
        # bottleneck in the overall calculation
        vals = sampler.draw_sample()
        if dofs is None:
            np.copyto(snapshot_array, vals)
        else:
            for (d, v) in zip(dofs, vals):
                snapshot_array[d] = v

    @staticmethod
    def _get_feature(snapshot_array, dofs):
        if dofs is None:
            return snapshot_array.ravel()
        else:
            return np.array([snapshot_array[i] for i in dofs]).ravel()


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
        self._fill_feature(snapshot_array=snapshot.coordinates,
                           sampler=self.coordinate_gaussian,
                           dofs=self.coordinate_dofs)
        self._fill_feature(snapshot_array=snapshot.momenta,
                           sampler=self.momentum_gaussian,
                           dofs=self.momentum_dofs)

    def __call__(self, snapshot):
        x_vals = self._get_feature(snapshot.coordinates,
                                   self.coordinate_dofs)
        p_vals = self._get_feature(snapshot.momenta,
                                   self.momentum_dofs)
        return (self.coordinate_gaussian(x_vals) 
                * self.momentum_gaussian(p_vals))


class MMSTElectronicGaussianInitialConditions(GaussianInitialConditions):
    __features__ = [features.electronic_coordinates,
                    features.electronic_momenta]
    @classmethod
    def with_n_dofs(cls, n_dofs):
        return cls(
            x0=np.array([0.0]*n_dofs), p0=np.array([0.0]*n_dofs), 
            alpha_x=np.array([1.0]*n_dofs), alpha_p=np.array([1.0]*n_dofs)
        )

    def generate_initial_snapshot(self, previous_snapshot):
        snapshot = previous_snapshot.copy() 
        # this might be a shallow copy, so deepen over the features:
        snapshot.electronic_momenta = snapshot.electronic_momenta.copy()
        snapshot.electronic_coordinates = snapshot.electronic_coordinates.copy()
        self.fill_initial_snapshot(snapshot, previous_snapshot)
        return snapshot


    def fill_initial_snapshot(self, snapshot, previous_snapshot):
        self._fill_feature(snapshot_array=snapshot.electronic_coordinates,
                           sampler=self.coordinate_gaussian,
                           dofs=self.coordinate_dofs)
        self._fill_feature(snapshot_array=snapshot.electronic_momenta,
                           sampler=self.momentum_gaussian,
                           dofs=self.momentum_dofs)


    def __call__(self, snapshot):
        x_vals = self._get_feature(snapshot.electronic_coordinates,
                                   self.coordinate_dofs)
        p_vals = self._get_feature(snapshot.electronic_momenta,
                                   self.momentum_dofs)
        return (self.coordinate_gaussian(x_vals) 
                * self.momentum_gaussian(p_vals))
