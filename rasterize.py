import numpy as np

from utils import get_raster_focal_directions, get_same_plane_index3, get_same_position_index3, foolog


def rasterize3(x, dx, intensity, x_camera, dx_camera, raster_dimension, raster_focal_directions=None):
    foolog('    creating rasters...')
    raster = np.zeros((raster_dimension, raster_dimension))
    if raster_focal_directions is None:
        raster_focal_directions = get_raster_focal_directions(raster_dimension, dx_camera)

    foolog('    Determine which rays don''t interact with the camera...')
    foolog('    (same position as camera)')
    same_position_index = get_same_position_index3(x, x_camera, dx_camera)
    foolog('    (same plane as camera)')
    same_plane_index = get_same_plane_index3(x, x_camera, dx_camera)
    foolog('    (combine)')
    mask = same_position_index * same_plane_index

    foolog('    reducing data...')
    foolog('    subset ray positions')
    x_subset = x[:, mask]
    foolog('    subset ray directions')
    dx_subset = dx[:, mask]
    foolog('    subset intensities')
    intensity_subset = intensity[mask]

    mask2 = (np.array([[1], [1], [1]], dtype=np.float32) - np.abs(dx_camera))
    x_reduced = x_subset[mask2.flatten() == 1] % 1

    foolog('    rasterizing {0} rays...'.format(x_reduced.shape[1]))

    for i in range(x_reduced.shape[1]):
        x = x_reduced[0][i]
        y = x_reduced[1][i]
        xx = int(x * raster_dimension)
        yy = int(y * raster_dimension)
        raster[yy][xx] += intensity_subset[i] * max(0, dx_subset[:, i].dot(raster_focal_directions[yy][xx]))

    return raster