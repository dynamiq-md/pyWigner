from openpathsampling.netcdfplus import StorableObject
import numpy as np


class Operator(StorableObject):
    def __init__(self):
        pass

    def sample_initial_conditions(self, previous_trajectory):
        raise NotImplementedError("No sampling function")

    def correction(self, snapshot):
        return 1.0

    def __call__(self, snapshot):
        raise NotImplementedError("No Wigner function for operator")


