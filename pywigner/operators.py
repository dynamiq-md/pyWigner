import openpathsampling as paths
import numpy as np

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
        pass

    def sample_initial_conditions(self, previous_trajectory):
        pass

    def correction(self, snapshot):
        pass

    def __call__(self, snapshot):
        pass

    def excite(dof, excitons=1):
        pass

