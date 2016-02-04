from pywigner.tools import clean_ravel
from pywigner.operators import Operator
import pywigner as lsc
import numpy as np

def real_multidim_gaussian(x, x0, gamma):
    """exp(-\sum_i gamma_i*(x_i-x0_i)^2)"""
    # TODO: note that this might be sped up by reusing expon instead of
    # re-alloc'ing.
    expon = numpy.sub(x-x0) # dx
    numpy.mul(expon, expon, expon) # dx^2
    numpy.mul(expon, gamma, expon) # gamma*dx^2
    return np.exp(-np.sum(expon))


class CoherentProjection(Operator):
    """Coherent projection operator: $|x_0 p_0; \gamma><x_0 p_0; \gamma|$.

    Technically, this will support not only coherent states, but all
    harmonic oscillator wavefunctions. Currently, we only allow at most 1
    exciton per dof. Excited states can either be set as a list of exciton
    counts by degree of freedom, or using the `.excite()` method.

    Attributes
    ----------
    x0 : numpy.array
    p0 : numpy.array
    gamma : numpy.array
    excitons : numpy.array
    inv_gamma : numpy.array
    position_sampling_width : numpy.array
    momentum_sampling_width : numpy.array
    """
    def __init__(self, x0, p0, gamma, excitons=0):
        # set gaussian parameters
        self.x0 = x0.ravel()
        self.p0 = p0.ravel()
        self.gamma = gamma.ravel()
        self.inv_gamma = 1.0 / gamma
        assert(len(self.x0) == len(self.p0))
        self.n_dofs = len(list(self.x0))

        self.gaussian_x = lsc.GaussianFunction(x0=self.x0, 
                                               alpha=self.gamma)
        self.gaussian_p = lsc.GaussianFunction(x0=self.p0,
                                               alpha=self.inv_gamma)

        # set the sampling_gamma
        # sampling_gamma = sampling_ratio[n_exciton]*gamma
        # sampling_inv_gamma = sampling_ratio[n_exciton]*inv_gamma
        self.exciton_sampling_ratios = {0 : 1.0, 1 : 1.1}
        self.sampler = self.default_sampler()

        # set up excitons
        self.excitons = clean_ravel(excitons, self.n_dofs)
        self._set_exciton_dict()


    def _set_exciton_dict(self):
        self._exciton_dict = {e.index() : e for e in self.excitons if e > 0}


    def default_sampler(self):
        alpha_x = [sampling_ratio[self.excitons[i]]*self.gamma[i]
                   for i in range(len(self.gamma))]
        alpha_p = [sampling_ratio[self.excitons[i]]*self.inv_gamma[i]
                   for i in range(len(self.inv_gamma))]
        return lsc.GaussianInitialConditions(x0=self.x0, p0=self.p0,
                                             alpha_x=alpha_x,
                                             alpha_p=alpha_p)


    def __call__(self, snapshot):
        return (self.gaussian_x(snapshot.coordinates)
                * self.gaussian_p(snapshot.momenta))

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
