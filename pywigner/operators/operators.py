from openpathsampling.netcdfplus import StorableObject
import numpy as np
import pywigner as lsc


class Operator(StorableObject):
    def __init__(self):
        self.sampler = None

    def __mul__(self, other):
        return ProductOperator([self, other])

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


class ProductOperator(Operator):
    def __init__(self, operators):
        super(ProductOperator, self).__init__()
        self.operators = operators
        # NOTE: we don't check any orthogonality here, but it will show up
        # if you try to use the default sampler.

    def __call__(self, snapshot):
        result = 1.0
        for op in self.operators:
            result *= op(snapshot)
        return result

    def __mul__(self, other):
        self.operators.append(other)

    def default_sampler(self):
        return lsc.samplers.OrthogonalInitialConditions(
            [op.default_sampler() for op in self.operators]
        )
