import numpy as np
import copy

from utils import foolog, random_unit_vector, create_random_vectors

HUGE_NUMBER = 2**31


def volume(x, dx):
    x_int = x // 1
    diff = (x_int == x) * np.minimum(0, np.sign(dx))
    return x_int + diff

def propagate(x0, dx0, intensity0, environment, mirror=True):
    """propagates set of N rays with position x0 and direction dx0 to x1, dx1.
    * x0: 3xN matrix
    * dx0: 3xN matrix
    * intensity: 1xN matrix
    * volumes: object that returns various properties of the space occupied by rays at position x
    """

    # take modulo to get relative position
    x = x0 % 1

    #foolog('    concatinate...')
    xx = np.concatenate([-1 - x, 0 - x, 1 - x], axis=0)
    dxx = np.concatenate([dx0, dx0, dx0], axis=0)

    # calculate "rate" needed to get to edges of cube for each axis
    #foolog('    division...')
    l = xx / dxx

    # remove nans/infs
    # remove 0 or negative numbers
    #foolog('    stuff...')
    l[(~np.isfinite(l)) | (l <= 0)] = HUGE_NUMBER

    # smallest rate of change gets chosen
    l_min = np.ndarray.min(l, axis=0)

    # edge case
    l_min[l_min == HUGE_NUMBER] = 1

    # new position
    #foolog('    new position...')
    x1 = x0 + l_min * dx0

    # given new position and old direction of a ray, we can determine the volume bordering the current one.
    # given the bordering volume, we want to know if it's solid or not.
    # if the bordering volume is solid, the ray should bounce back (i.e. flip direction w.r.t. normal of the solid)
    # if not, the ray direction should remain the same

    # bordering volume
    #foolog('    volume diff...')
    current_volume = volume(x0, dx0)
    bordering_volume = volume(x1, dx0)
    position_diff = bordering_volume - current_volume

    #foolog('    calculate remaining intensity...')
    solid = environment.get_solid(bordering_volume.astype(int)) # is bordering volume solid?
    albedo = environment.get_albedo(bordering_volume.astype(int)) # what is the bordering volume's albedo?
    absorption = (1 - albedo) * solid
    intensity1 = intensity0 * (1 - absorption)

    #foolog('    change direction...')
    # only need to change direction if the change in position coincides with a solid bordering volume
    change_direction = position_diff * solid
    change_direction_index = np.sum(np.abs(change_direction), axis=0)
    # if bordering volume is solid, change direction; else direction stays the same

    if mirror:
        dx1 = (1 - 2 * np.abs(change_direction)) * dx0
    else:
        dx1 = random_direction_change(dx0, change_direction_index.astype(bool), -change_direction)

    #rasterize(x1, dx1, intensity1)

    #sign_ok = np.sign(dx_mirror) * np.sign(dx1)
    return x1, dx1, intensity1


def random_direction_change(dx, change_direction_index, change_direction):
    result = copy.copy(dx)
    normals = change_direction[:, change_direction_index]
    if np.any(change_direction_index):
        result[:, change_direction_index] = create_random_vectors(normals)
    return result
