import numpy as np
from openmodelica_microgrid_gym.util import RandProcess


class RandProcessWrapper(RandProcess):

    def reset(self, initial=None):
        """
        Resets the process, if initial is None, it is set randomly in the range of bounds
        """
        if initial is None:
            self._last = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
            self.proc.mean = self._last
            # self.reserve = self._last
        else:
            self._last = initial
            self.proc.mean = self._last
            #self.reserve = self._last
        self._last_t = 0
        self._reserve = None



