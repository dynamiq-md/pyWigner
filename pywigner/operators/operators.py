from openpathsampling.netcdfplus import StorableObject
import numpy as np
import pywigner as lsc


class Operator(StorableObject):
    def __init__(self):
        self.sampler = None

    def correction(self, snapshot, sampler):
        """ Op.correction(snapshot) = Op(snapshot) / Op.sampler(snapshot)

        This is the correction to give us the time-independent contribution
        to the integrand when this operator is the A-operator, based on
        using `self.sampler` as the sampling approach.

        Operators can override this to use shortcuts with various types of
        samplers.
        """
        retval = self(snapshot) / sampler(snapshot) * sampler.norm
        # TODO: add checks for TypeError if self.sampler isn't callable?
        return retval

    def __call__(self, snapshot):
        raise NotImplementedError("No Wigner function for abstract operator")

    def default_sampler(self):
        raise NotImplementedError("No default sampler for abstract operator")


class OrthogonalProductOperator(Operator):
    def __init__(self, operators):
        super(OrthogonalProductOperator, self).__init__()
        self.operators = operators
        # TODO: check that the operators really are othogonal? Or leave that
        # to the sampler?

    def __call__(self, snapshot):
        result = 1.0
        for op in self.operators:
            result *= op(snapshot)
        return result

    def default_sampler(self):
        return lsc.samplers.OrthogonalInitialConditions(
            [op.default_sampler() for op in self.operators]
        )
