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
    """
    Coherent projection operator: :math:`|x_0 p_0; \gamma><x_0 p_0; \gamma|`.

    Technically, this will support not only coherent states, but all
    harmonic oscillator wavefunctions. Currently, we only allow at most 1
    exciton per dof. Excited states can either be set as a list of exciton
    counts by degree of freedom, or using the `.excite()` method.

    Notes
    -----
    We define the width of the coherent state such that

    .. math:: 
        <x|x_0 p_0; \gamma> 
        = \frac{\gamma}{\pi} 
          \exp\(-\frac{\gamma}{2}(x - x_0)^2 + i p_0 (x-x_0)\)

    TODO: check sign of the imaginary part

    This gives us the Wigner function:
    
    .. math::
        (|x_0 p_0; \gamma><x_0 p_0; \gamma|)_W(q, p)
        =
    
    TODO: fill this in


    Attributes
    ----------
    x0 : numpy.array
    p0 : numpy.array
    gamma : numpy.array
    dofs : list
    excitons : list
    """
    def __init__(self, x0, p0, gamma, dofs=None, excitons=0):
        # set gaussian parameters
        self.x0 = x0.ravel()
        self.p0 = p0.ravel()
        self.gamma = gamma.ravel()
        self.inv_gamma = 1.0 / self.gamma
        self.dofs = dofs
        if self.dofs is None:
            self.n_dofs = len(list(self.x0))
        else:
            self.n_dofs = len(dofs)
        assert(len(self.x0) == len(self.p0))
        assert(len(self.gamma) == len(self.x0))
        assert(len(self.p0) == self.n_dofs)

        self.gaussian_x = lsc.tools.GaussianFunction(x0=self.x0, 
                                                     alpha=self.gamma)
        self.gaussian_p = lsc.tools.GaussianFunction(x0=self.p0,
                                                     alpha=self.inv_gamma)

        # set the sampling_gamma
        # sampling_gamma = sampling_ratio[n_exciton]*gamma
        # sampling_inv_gamma = sampling_ratio[n_exciton]*inv_gamma
        self.exciton_sampling_ratios = {0 : 1.0, 1 : 1.1}

        # set up excitons
        self.excitons = clean_ravel(excitons, self.n_dofs)
        self._set_exciton_dict()


    def _set_exciton_dict(self):
        self._exciton_dict = {e.index() : e for e in self.excitons if e > 0}

    @staticmethod
    def _get_feature(feature_array, dofs):
        if dofs is None:
            return feature_array.ravel()
        else:
            return np.array([feature_array.ravel()[i] for i in dofs])

    def default_sampler(self, exciton_sampling_ratios=None):
        if exciton_sampling_ratios is None:
            exciton_sampling_ratios = self.exciton_sampling_ratios
        #TODO: check that these gamma->alpha setups are correct
        alpha_x = [exciton_sampling_ratios[self.excitons[i]]*self.gamma[i]
                   for i in range(len(self.gamma))]
        alpha_p = [exciton_sampling_ratios[self.excitons[i]]*self.inv_gamma[i]
                   for i in range(len(self.inv_gamma))]
        return lsc.samplers.GaussianInitialConditions(
            x0=self.x0, alpha_x=alpha_x, coordinate_dofs=self.dofs,
            p0=self.p0, alpha_p=alpha_p, momentum_dofs=self.dofs
        )


    def __call__(self, snapshot):
        x_vals = self._get_feature(snapshot.coordinates.ravel(), self.dofs)
        p_vals = self._get_feature(snapshot.momenta.ravel(), self.dofs)
        # TODO: correct for excite; check for normalization
        return self.gaussian_x(x_vals) * self.gaussian_p(p_vals)

    def excite(self, dof, excitons=1):
        try:
            paired = zip(dof, excitons)
        except TypeError:
            paired = [(dof, excitons)]
        for (d, e) in paired:
            self.excitons[d] = e
        self._set_exciton_dict()

class ElectronicCoherentProjection(CoherentProjection):
    @classmethod
    def with_n_dofs(cls, n_dofs):
        return cls(
            x0=np.array([0.0]*n_dofs), 
            p0=np.array([0.0]*n_dofs), 
            gamma=np.array([0.0]*n_dofs)
        )

    def __call__(self, snapshot):
        return (self.gaussian_x(snapshot.electronic_coordinates.ravel())
                * self.gaussian_p(snapshot.electronic_momenta.ravel()))

    def default_sampler(self):
        return lsc.samplers.MMSTElectronicGaussianInitialConditions.with_n_dofs(2)
