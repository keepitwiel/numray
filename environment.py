import numpy as np

class Environment(object):
    def __init__(self):
        self._solid = np.ones((10, 10, 10), dtype=np.float32)
        self._solid[1:9, 1:9, 1:9] = 0 # inside volume is not solid
        self._solid[1:9, 7:9, 6] = 1 # a ridge protruding from wall opposite the camera
        self._default_solid = 0

        self._albedo = 0.8 * np.ones((10, 10, 10), dtype=np.float32)
        # self._albedo[0:9, 0:9, 9] = 0 # light source has 0 albedo
        # self._albedo[0:9, 0:9, 0] = 1.0 # floor has very high albedo
        self._default_albedo = 0.5

    def get_solid(self, x: np.ndarray) -> np.ndarray:
        """Returns an array indicating whether the volumes at given positions are solid or not.
        Input:
        x: 3xN integer array: indicates 3-dimensional positions of N volumes. This array will be sliced so
        the output can be properly generated."""

        return self._solid[x[0], x[1], x[2]]

    def get_albedo(self, x: np.ndarray) -> np.ndarray:
        """Returns an array indicating the albedo of the volumes at given positions.
        Input:
        x: 3xN integer array: indicates 3-dimensional positions of N volumes. This array will be sliced so
        the output can be properly generated."""

        return self._albedo[x[0], x[1], x[2]]
