import numpy as np
import time
import png
import click
from distutils.util import strtobool

from environment import Environment
from utils import generate_rays_from_light_source, get_raster_focal_directions, foolog
from propagate import propagate
from rasterize import rasterize3

RAYS_PER_SIDE = 500
NR_OF_PROPAGATIONS = 40
RASTER_DIMENSION = 40

@click.command()
@click.option('--nr_of_propagations', default=NR_OF_PROPAGATIONS, help='Number of times a set of rays is being propagated.')
@click.option('--rays_per_side', default=RAYS_PER_SIDE, help='Number of rays per side of the light source square.')
@click.option('--raster_dimension', default=RASTER_DIMENSION, help='Number of horizontal/vertical pixels in the camera raster.')
@click.option('--mirror', default=True, help='Reflection is mirror-like (default) or random.')
@click.option('--normalize_brightness', default=True, help='Normalizes brightness in the output image.')
def main(rays_per_side, nr_of_propagations, raster_dimension, mirror, normalize_brightness):
    mirror = bool(strtobool(mirror))
    normalize_brightness = bool(strtobool(normalize_brightness))

    t0 = time.time()

    foolog('generating rays...')
    #x, dx, intensity = generate_rays(rays_per_side)
    dx_source = np.array([[0], [0], [-1]])

    x, dx, intensity = generate_rays_from_light_source(2, 8, 2, 8, 9, dx_source, True, rays_per_side)

    environment = Environment()
    x_camera = np.array([[3], [1], [3]])
    dx_camera = np.array([[0], [-1], [0]])
    focal_directions = get_raster_focal_directions(raster_dimension, dx_camera)
    raster = None

    foolog('starting ray propagation...')
    for r in range(nr_of_propagations+1):
        foolog('Step {0}. Remaining average intensity: {1:2.3f}'.format(r, np.average(intensity)))
        foolog('Average position: {0}'.format(np.average(x, axis=1)))

        foolog('rasterize...')
        result = rasterize3(x, dx, intensity, x_camera, dx_camera, RASTER_DIMENSION, focal_directions)
        if raster is None:
            raster = result
        else:
            raster += result

        foolog('    output to png...')

        brightness_factor = 1
        if normalize_brightness:
            m = np.max(raster)
            if m > 0:
                brightness_factor = 255.0 / m

        to_png = (brightness_factor * raster).astype(np.uint8)
        png.from_array(to_png, 'L').save('output/output_n={0}.png'.format(r))

        if r < nr_of_propagations:
            foolog('propagate...')
            x, dx, intensity = propagate(x, dx, intensity, environment, mirror=mirror)

    t1 = time.time()
    foolog('total time: {0:2.3f} seconds'.format(t1 - t0))

if __name__ == '__main__':
    main()
