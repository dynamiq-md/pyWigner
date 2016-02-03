from pywigner.tools import clean_ravel
from pywigner.operators import Operator
import numpy as np

class CoherentProjection(Operator):
    """Coherent projection operator: $|x_0 p_0; \gamma><x_0 p_0; \gamma|$.

    Technically, this will support not only coherent states, but all
    harmonic oscillator wavefunctions. Currently, we only allow at most 1
    exciton per dof. Excited states can either be set as a list of exciton
    counts by degree of freedom, or using the `.excite()` method.
    """
    def __init__(self, x0, p0, gamma, excitons=0):
        self.x0 = x0.ravel()
        self.p0 = p0.ravel()
        self.gamma = gamma.ravel()
        assert(len(self.x0) == len(self.p0))
        self.n_dofs = len(list(self.x0))
        self.excitons = clean_ravel(excitons, self.n_dofs)
        if list(self.excitons) == [0]*self.n_dofs:
            self.sampling_gamma = self.gamma
        else:
            # TODO: set up things beyond the ground state
            pass
        self._set_exciton_dict()

    def _set_exciton_dict(self):
        self._exciton_dict = {e.index() : e for e in self.excitons if e > 0}

    def sample_initial_conditions(self, previous_trajectory, new_snapshot):
        if (list(self.sampling_gamma) == list(self.gamma) 
                and list(self.excitons) == [0]*self.n_dofs):
            pass
        else:
            pass

    def sampling_function(self, snapshot):
        pass

    def correction(self, snapshot):
        """sample_initial_conditions * correction == __call__"""
        if (list(self.sampling_gamma) == list(self.gamma) 
                and list(self.excitons) == [0]*self.n_dofs):
            # sampling function *is* operator
            return 1.0
        else:
            return self(snapshot) / self.sampling_function(snapshot)

    def __call__(self, snapshot):
        pass

    def excite(self, dof, excitons=1):
        try:
            paired = zip(dof, excitons)
        except TypeError:
            paired = [(dof, excitons)]
        for (d, e) in paired:
            self.excitons[d] = e
        self._set_exciton_dict()

class ElectronicCoherentProjection(CoherentProjection):
    def __call__(self, snapshot):
        pass

    def sampling_function(self, snapshot):
        pass
