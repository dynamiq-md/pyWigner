import openpathsampling as paths
import numpy as np

# TODO: move to tools
def clean_ravel(arr, n_dofs):
    try:
        retval = arr.ravel()
    except AttributeError:
        retval = [arr] * n_dofs
    return retval

class Operator(paths.OPSNamed):
    def __init__(self):
        pass

    def sample_initial_conditions(self, previous_trajectory):
        raise NotImplementedError("No sampling function")

    def correction(self, snapshot):
        return 1.0

    def __call__(self, snapshot):
        raise NotImplementedError("No Wigner function for operator")


class GaussianWavepacket(Operator):
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

    def excite(dof, excitons=1):
        pass

