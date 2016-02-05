import pywigner as lsc

class InitialConditionSampler(object):

    global_engine = None

    def __init__(self):
        self._engine = None

    def prepare(self, n_frames=None, engine=None):
        if engine is not None:
            self.engine = engine
        if n_frames is not None:
            self.n_frames = n_frames

    @property
    def engine(self):
        if self._engine is not None:
            return self._engine
        elif self.global_engine is not None:
            return self.global_engine
        else:
            raise RuntimeError("Can't create trajectory: no engine set.")

    @engine.setter
    def engine(self, value):
        self._engine = value

    def generate_trial_trajectory(self, previous_trajectory):
        snap0 = self.generate_initial_snapshot(previous_trajectory[0])
        # TODO
        pass

    def generate_initial_snapshot(self, previous_snapshot):
        raise NotImplementedError("Abstract InitialConditionSampler")

    def __call__(self, snapshot):
        raise NotImplementedError("Abstract InitialConditionSampler")
        pass

class OrthogonalProductInitialConditions(InitialConditionSampler):
    def __init__(self, samplers):
        pass

    def generate_trial_trajectory(self, previous_trajectory):
        pass

    def __call__(self, snapshot):
        pass
