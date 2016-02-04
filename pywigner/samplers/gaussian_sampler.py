import pywigner as lsc

class GaussianInitialConditions(lsc.InitialConditionSampler):
    def __init__(self, x0, p0, alpha_x, alpha_p):
        pass

    def generate_trial_trajectory(self, previous_trajectory):
        pass

    def __call__(self, snapshot):
        pass

class MMSTElectronicGaussianInitialConditions(GaussianInitialConditions):
    @classmethod
    def with_n_dofs(cls, n_dofs):
        return MMSTElectronicGaussianSampler(
            x0=np.array([0.0]*n_dofs), p0=np.array([0.0]*n_dofs), 
            alpha_x=np.array([1.0]*n_dofs), alpha_p=np.array([1.0]*n_dofs)
        )

    def __call__(self, snapshot):
        pass

    def generate_trial_trajectory(self, previous_trajectory):
        pass
