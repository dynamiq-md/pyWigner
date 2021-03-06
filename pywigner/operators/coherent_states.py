from pywigner.tools import clean_ravel
from pywigner.operators import Operator
import pywigner as lsc
import dynamiq_samplers as samplers
import numpy as np

def raveled_numpyify(arr):
    try:
        retval = arr.ravel()
    except AttributeError:
        retval = np.array(arr).ravel()
    return retval

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


    This gives us the Wigner function:
    
    .. math::
        (|x_0 p_0; \gamma><x_0 p_0; \gamma|)_W(q, p)
        = ??? \exp(-\gamma (q-x_0)^2 - 1/\gamma (p-p_0)^2

    TODO: fill in the norm correctly

    Of course, the total operator is the product of this for each degree of
    freedom.

    Using excited state wavepackets (for which we assume that the coherent
    state is the ground state of some harmonic oscillator, and then we
    excite a mode) gives a very simple correction to this. So far, the code
    only supports the first excitation, where the correction is

    .. math::
        (|x_0 p_0; \gamma, n=1><x_0 p_0; \gamma, n=1|)_W(q, p) =
        D_1(q, p) (|x_0 p_0; \gamma><x_0 p_0; \gamma|)_W(q, p)

    where

    .. math::
        D_1(q, p) = ???
        


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
        self.x0 = raveled_numpyify(x0)
        self.p0 = raveled_numpyify(p0)
        self.gamma = raveled_numpyify(gamma)
        self.inv_gamma = 1.0 / self.gamma
        self.dofs = dofs
        if self.dofs is None:
            self.n_dofs = len(list(self.x0))
        else:
            self.n_dofs = len(dofs)

        # sanity checks for bad input
        assert(len(self.x0) == len(self.p0))
        assert(len(self.gamma) == len(self.x0))
        assert(len(self.p0) == self.n_dofs)

        self.gaussian_x = samplers.tools.GaussianFunction(x0=self.x0, 
                                                          alpha=self.gamma)
        self.gaussian_p = samplers.tools.GaussianFunction(x0=self.p0,
                                                          alpha=self.inv_gamma)

        # This sets the total factor outside the Wigner function to 1.0,
        # since the norms of the gaussians otherwise carry. Note that we're
        # undoing doing some extra multiplications here; needed to keep
        # `correction` in line -- but there might be a better way.
        # TODO: is this actually correct?
        self.norm = np.prod([2.0]*self.n_dofs)
        self.norm *= 1.0/(self.gaussian_x.norm * self.gaussian_p.norm)

        # set the sampling_gamma
        # sampling_gamma = sampling_ratio[n_exciton]*gamma
        # sampling_inv_gamma = sampling_ratio[n_exciton]*inv_gamma
        self.exciton_sampling_ratios = {0 : 1.0, 1 : 1.1}

        # set up excitons: must be done AFTER setting self.n_dofs
        self.excitons = excitons

    @property
    def excitons(self):
        return self._excitons

    @excitons.setter
    def excitons(self, val):
        self._excitons = clean_ravel(val, self.n_dofs)
        self._exciton_dict = {i : self._excitons[i] 
                              for i in range(len(self._excitons)) 
                              if self._excitons[i] > 0}

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
        return samplers.GaussianInitialConditions(
            x0=self.x0, alpha_x=alpha_x, coordinate_dofs=self.dofs,
            p0=self.p0, alpha_p=alpha_p, momentum_dofs=self.dofs
        )

    def excite(self, dof, excitons=1):
        try:
            paired = zip(dof, excitons)
        except TypeError:
            paired = [(dof, excitons)]

        old_excitons = self.excitons
        for (d, e) in paired:
            old_excitons[d] = e
        self.excitons = old_excitons
        return self

    def _call_excited_part(self, x_vals, p_vals):
        result = 1.0
        for (i, n) in self._exciton_dict.items():
            x_i = x_vals[i] - self.x0[i]
            p_i = p_vals[i] - self.p0[i]
            if n == 1:
                result *= 2.0*(x_i*x_i + p_i*p_i - 0.5)
        return result

    def __call__(self, snapshot):
        x_vals = self._get_feature(snapshot.coordinates.ravel(), self.dofs)
        p_vals = self._get_feature(snapshot.momenta.ravel(), self.dofs)

        # TODO: correct for excite; check for normalization
        standard_part = self.gaussian_x(x_vals) * self.gaussian_p(p_vals)
        excited_part = self._call_excited_part(x_vals, p_vals)
        result = self.norm * standard_part * excited_part
        return result


class ElectronicCoherentProjection(CoherentProjection):
    @classmethod
    def with_n_dofs(cls, n_dofs):
        return cls(
            x0=np.array([0.0]*n_dofs), 
            p0=np.array([0.0]*n_dofs), 
            gamma=np.array([1.0]*n_dofs)
        )

    def __call__(self, snapshot):
        x_vals = self._get_feature(snapshot.electronic_coordinates.ravel(), 
                                   self.dofs)
        p_vals = self._get_feature(snapshot.electronic_momenta.ravel(), 
                                   self.dofs)

        # TODO: correct for excite; check for normalization
        standard_part = self.gaussian_x(x_vals) * self.gaussian_p(p_vals)
        excited_part = self._call_excited_part(x_vals, p_vals)
        result = self.norm * standard_part * excited_part
        return result

    def default_sampler(self, exciton_sampling_ratios=None):
        if exciton_sampling_ratios is None:
            exciton_sampling_ratios = self.exciton_sampling_ratios
        #TODO: check that these gamma->alpha setups are correct
        alpha_x = [exciton_sampling_ratios[self.excitons[i]]*self.gamma[i]
                   for i in range(len(self.gamma))]
        alpha_p = [exciton_sampling_ratios[self.excitons[i]]*self.inv_gamma[i]
                   for i in range(len(self.inv_gamma))]
        return samplers.MMSTElectronicGaussianInitialConditions(
            x0=self.x0, alpha_x=alpha_x, coordinate_dofs=self.dofs,
            p0=self.p0, alpha_p=alpha_p, momentum_dofs=self.dofs
        )
