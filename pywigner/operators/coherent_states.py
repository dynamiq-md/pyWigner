from ..tools import clean_ravel
from . import Operator

class CoherentProjection(Operator):
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

    def sample_initial_conditions(self, previous_trajectory):
        pass

    def correction(self, snapshot):
        if list(self.sampling_gamma) == list(self.gamma):
            if list(self.excitons) == [0]*self.n_dofs:
                return 1.0
            else:
                pass
        else:
            pass

    def __call__(self, snapshot):
        pass

    def excite(self, dof, excitons=1):
        pass

