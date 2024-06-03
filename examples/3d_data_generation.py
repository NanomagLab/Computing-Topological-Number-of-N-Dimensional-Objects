import numpy as np
"""ball"""
gird_size = 200
radius = 0.7
x, y, z = np.meshgrid(
    np.linspace(-1, 1, gird_size),
    np.linspace(-1, 1, gird_size),
    np.linspace(-1, 1, gird_size)
)
r = x ** 2. + y ** 2. + z ** 2.
arr = np.greater(radius, r).astype(np.float32) * 2. - 1.
np.save("3d/ball.npy", arr)

"""shell"""
gird_size = 200
inner_radius = 0.6
outer_radius = 0.8
x, y, z = np.meshgrid(
    np.linspace(-1, 1, gird_size),
    np.linspace(-1, 1, gird_size),
    np.linspace(-1, 1, gird_size)
)
r = x ** 2. + y ** 2. + z ** 2.
arr = (np.greater(r, inner_radius) * np.greater(outer_radius, r)).astype(np.float32) * 2. - 1.
np.save("3d/shell.npy", arr)

"""doughnut"""
gird_size = 200
major_radius = 0.4
minor_radius = 0.2
x, y, z = np.meshgrid(
    np.linspace(-1, 1, gird_size),
    np.linspace(-1, 1, gird_size),
    np.linspace(-1, 1, gird_size)
)


def cartesian_to_toroidal(x, y, z, R):
    theta = np.arctan2(y, x)
    d = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt((d - R) ** 2 + z ** 2)
    phi = np.arctan2(z, d - R)
    return theta, phi, r


theta, phi, r = cartesian_to_toroidal(x, y, z, major_radius)
arr = np.greater(minor_radius, r).astype(np.float32) * 2. - 1.
np.save("3d/doughnut.npy", arr)
