import numpy as np
import time
from datetime import datetime

ASCII_MAP = ' .,:ilwW'


def create_random_vectors(source_vectors):
    vector_list = [random_unit_vector(source_vectors[:, i]) for i in range(source_vectors.shape[1])]
    result = np.concatenate(vector_list, axis=1)
    #check = source_vectors * result
    return result


def random_unit_vector(n):
    if len(n.shape)==1:
        n = np.expand_dims(n, axis=2)

    #-0.5, 0, 0.866
    # [1, 0, 0] -> 0.5, 0, 0.866
    # [-1, 0, 0] -> -0.5, 0, 0.866

    phi = np.random.uniform(-np.pi, np.pi)
    theta = np.random.uniform(0, 0.5 * np.pi)
    result = np.array([[np.cos(theta) * np.cos(phi)], [np.cos(theta) * np.sin(phi)], [np.sin(theta)]])
    result = np.sign(n) * np.abs(result) + (n == 0) * result
    return result


def generate_light_source_grid(x0, y0, x1, y1, z, rays_per_side):
    x = np.array([n for _ in np.linspace(x0, x1, rays_per_side) for n in np.linspace(x0, x1, rays_per_side)],
                 dtype=np.float32)
    y = np.array([n for n in np.linspace(y0, y1, rays_per_side) for _ in np.linspace(y0, y1, rays_per_side)],
                 dtype=np.float32)
    z = 9 * np.ones(rays_per_side ** 2, dtype=np.float32)

    return np.concatenate([[x], [y], [z]], axis=0)


def generate_rays_from_light_source(x0, y0, x1, y1, z, dx, random, rays_per_side):
    x = generate_light_source_grid(x0, y0, x1, y1, z, rays_per_side)

    if random:
        dxx = np.concatenate([random_unit_vector(dx) for _ in range(x.shape[1])], axis=1)
    else:
        dxx = np.concatenate([dx for _ in range(x.shape[1])], axis=1)

    intensity = np.ones(x.shape[1])

    return x, dxx, intensity


def generate_rays(rays_per_side):
    general_direction = np.array([[0.5], [0], [-0.5*np.sqrt(3)]])

    x = np.array([n for _ in np.linspace(2, 8, rays_per_side) for n in np.linspace(2, 8, rays_per_side)],
                 dtype=np.float32)
    y = np.array([n for n in np.linspace(2, 8, rays_per_side) for _ in np.linspace(2, 8, rays_per_side)],
                 dtype=np.float32)
    z = 9 * np.ones(rays_per_side ** 2, dtype=np.float32)

    xx = np.concatenate([[x], [y], [z]], axis=0)
    dxx = np.concatenate([random_unit_vector(general_direction) for _ in range(rays_per_side ** 2)], axis=1)
    #dxx = np.concatenate([general_direction for _ in range(rays_per_side ** 2)], axis=1)
    intensity = np.ones(rays_per_side**2)

    return xx, dxx, intensity


def get_same_position_index3(x, x_camera, dx_camera):
    mask = (np.array([[1], [1], [1]], dtype=np.float32) - np.abs(dx_camera)).flatten()
    x_reduced = x[mask == 1, :]
    x_to_compare = np.multiply.outer(x_camera[mask == 1], np.ones(x.shape[1:3])).squeeze()
    return np.product((x_reduced // 1) == x_to_compare, axis=0, dtype=np.bool)


def get_same_plane_index3(x, x_camera, dx_camera):
    mask = np.abs(dx_camera).flatten()
    x_reduced = x[mask == 1, :]
    x_to_compare = np.multiply.outer(x_camera[mask == 1], np.ones(x.shape[1:3])).squeeze()
    return (x_reduced == x_to_compare).squeeze()


def get_raster_focal_directions(raster_dimension, dx):
    idx = np.array([1, 1, 1], dtype=np.float32) - np.abs(dx.flatten()) == True
    matrix = np.concatenate([np.identity(3)[:, idx], dx], axis=1)

    result = np.zeros((raster_dimension, raster_dimension, 3, 1))

    for x in range(raster_dimension):
        xx = 0.5 - ((x + 0.5) / raster_dimension)
        for y in range(raster_dimension):
            yy = 0.5 - ((y + 0.5) / raster_dimension)
            v = np.array([[xx], [yy], [1 - np.sqrt(xx**2 + yy**2)]])
            result[y][x] = matrix.dot(v)

    return result


def print_raster(raster):
    # return str(self._raster)
    a = np.minimum(len(ASCII_MAP) - 1, raster)
    s = ''
    for y in range(RASTER_DIMENSION):
        for x in range(RASTER_DIMENSION):
            s += 2 * ASCII_MAP[int(a[y][x])]
        s += '\n'

    return s


def foolog(text):
    print('{0} - {1}'.format(datetime.fromtimestamp(time.time()).isoformat(), text))