import numpy as np

class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, t):
        if self.decay in ["linear"]:
            out = max(self.finish, self.start - self.delta * t)
        elif self.decay in ["exp"]:
            out = min(self.start, max(self.finish, np.exp(- t / self.exp_scaling)))
        return out
    pass