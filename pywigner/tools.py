import numpy as np

def clean_ravel(arr, n_dofs):
    try:
        retval = arr.ravel()
    except AttributeError:
        retval = [arr] * n_dofs
    return retval


class GaussianFunction(object):
    def __init__(self, x0, alpha):
        self.x0 = np.array(x0)
        self.alpha = np.array(alpha)
        self.sigma = 1.0/np.sqrt(2.0*self.alpha)
        self.norm = np.prod(np.sqrt(self.alpha / np.pi))
        assert(self.x0.shape == self.alpha.shape)
        self._internal = np.zeros_like(self.alpha)

    def draw_sample(self):
        # creates array `sample` and returns it as a properly drawn sample
        sample = np.zeros_like(self._internal)
        self.set_array_to_drawn_sample(sample)
        return sample

    def set_array_to_drawn_sample(self, array):
        # TODO: I think this can be significantly sped up for large systems
        n_dofs = len(self.sigma)
        for i in range(n_dofs):
            array[i] = np.random.normal(loc=self.x0[i], scale=self.sigma[i])

    def __call__(self, x):
        np.subtract(x, self.x0, self._internal) # dx
        np.multiply(self._internal, self._internal, self._internal) # dx^2
        np.multiply(self._internal, self.alpha, self._internal) # alpha*dx^2
        return np.exp(-np.sum(self._internal))*self.norm
