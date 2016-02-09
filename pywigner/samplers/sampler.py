import pywigner as lsc

class InitialConditionSampler(object):
    __features__ = []

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

class OrthogonalInitialConditions(InitialConditionSampler):
    def __init__(self, samplers):
        self.samplers = samplers
        self.__features__ = list(set(sum(
            [s.__features__ for s in self.samplers], []
        )))

        feature_samplers = {f : [s for s in samplers if f in s.__features__]
                            for f in self.__features__}
        self.feature_dofs = {}
        for f in feature_samplers:
            if len(feature_samplers[f]) > 1:
                all_dofs = sum(list(feature_samplers[f].feature_dofs), [])
                if None in all_dofs or len(all_dofs) != len(set(all_dofs)):
                    raise RuntimeError("Some dofs repeated for feature "+
                                       str(f))
            else:
                all_dofs = feature_samplers[f].feature_dofs
            self.feature_dofs[f] = all_dofs



    def generate_initial_snapshot(self, previous_snapshot):
        pass

    def fill_initial_snapshot(self, snapshot, previous_snapshot):
        pass

    def __call__(self, snapshot):
        pass
