import numpy as np
import png

from utils import get_raster_focal_directions, foolog

class Camera(object):

    def __init__(self, x, dx, raster_dimension):
        self._x = x
        self._dx = dx
        self._dimension = raster_dimension
        self._raster = np.zeros((raster_dimension, raster_dimension))
        self._raster_focal_directions = get_raster_focal_directions(raster_dimension, dx)
        self._idx = np.where(True - np.abs(dx))[0] # result should be 1 dimensional array of length 2

    def rasterize(self, x_rel, x_abs, dx, intensity):
        n = x_rel.shape[1]

        # check if camera position is same as ray position
        # TODO: np.ones shouldn't have to be called every time - think of something better
        same_position = (x_abs == self._x)
        same_position_index = np.product(same_position, axis=0, dtype=np.bool)

        # check if normal is lined up with camera direction

        # example:
        zero = (x_rel == 0)
        normal = zero * np.sign(dx)
        alignment_with_camera_index = (self._dx.T.dot(normal)) == 1
        # dx = (0, 0, -1); x_rel = (0.5, 0, 0.5); -> (0, 0, -1) * (F, T, F) = (0, 0, 0)
        # dx = (0, 0, -1); x_rel = (0, 0.5, 0) -> (0, 0, -1) * (T, F, T) = (0, 0, -1)
        # dx = (0, 0, 1); x_rel = (0, 0.5, 0) -> (0, 0, 1) * (T, F, T) = (0, 0, 1)

        mask = (same_position_index * alignment_with_camera_index).flatten()

        if np.any(mask):
            x_subset = x_rel[:, mask]
            dx_subset = dx[:, mask]
            intensity_subset = intensity[mask]

            #foolog('    rasterizing {0} rays... (new)'.format(x_subset.shape[1]))

            for i in range(x_subset.shape[1]):
                x = x_subset[self._idx[0]][i]
                y = x_subset[self._idx[1]][i]
                xx = int(x * self._dimension)
                yy = int(y * self._dimension)
                self._raster[yy][xx] += intensity_subset[i] * max(0, dx_subset[:, i].dot(self._raster_focal_directions[yy][xx]))

    def to_png(self, epoch, normalize_brightness):
        brightness_factor = 1
        if normalize_brightness:
            m = np.max(self._raster)
            if m > 0:
                brightness_factor = 255.0 / m

        output_raster = (brightness_factor * self._raster).astype(np.uint8)
        png.from_array(output_raster, 'L').save('output/output_epoch={0}.png'.format(epoch))
