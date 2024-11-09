"""
This file contains the implementation for processing data of arbitrary dimensions.
The code was developed during our research process and may be more complex and harder to understand.
It is recommended for advanced users who need to work with data of arbitrary dimensions.
"""
from typing import Tuple

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, gamma
from itertools import permutations


def periodic_padding(arr, mode='bi'):
    if mode == "bi":
        return periodic_padding_bidirectional(arr)
    elif mode == "uni":
        return periodic_padding_unidirectional(arr)
    else:
        raise


def periodic_padding_unidirectional(arr):
    """
    :param
    arr: (size_1, size_2, ..., size_N, channels)
    :return:
    (size_1 + 1, size_2 + 1, ..., size_N + 1, channels)
    """
    N = len(tf.shape(arr)) - 1
    padded = arr
    for M in range(N):
        index = [slice(None)] * N
        index[M] = slice(0, 1)
        padded = tf.concat([
            padded,
            padded[tuple(index)]
        ], axis=M)
    return padded


def periodic_padding_bidirectional(arr):
    """
    :param
    arr: (size_1, size_2, ..., size_N, channels)
    :return:
    (size_1 + 2, size_2 + 2, ..., size_N + 2, channels)
    """
    N = len(tf.shape(arr)) - 1
    arr = arr
    for M in range(N):
        index_1 = [slice(None)] * N
        index_1[M] = slice(-1, None)
        index_2 = [slice(None)] * N
        index_2[M] = slice(0, 1)
        arr = tf.concat([
            arr[tuple(index_1)],
            arr,
            arr[tuple(index_2)]
        ], axis=M)
    return arr


def neighbor_sum(arr):
    """
    :param arr:
    (size_1, size_2, ..., size_N, channels)
    :return:
    (size_1, size_2, ..., size_N, channels)
    """
    summed_neigbors = tf.zeros_like(arr)
    arr = periodic_padding(arr)
    N = len(tf.shape(arr)[:-1])
    for M in range(N):
        index_1 = [slice(1, -1)] * N
        index_1[M] = slice(2, None)
        index_2 = [slice(1, -1)] * N
        index_2[M] = slice(0, -2)
        summed_neigbors = summed_neigbors + arr[tuple(index_1)] + arr[tuple(index_2)]
    return summed_neigbors


def gradient(arr):
    """
    :param arr:
    (size_1, size_2, ..., size_N, 1)
    :return:
    (size_1, size_2, ..., size_N, N)
    """
    assert tf.shape(arr)[-1] == 1
    N = len(tf.shape(arr)[:-1])
    grad = tf.zeros_like(arr)[..., :0]
    arr = periodic_padding(arr)
    for M in range(N):
        index_1 = [slice(1, -1)] * N
        index_1[M] = slice(2, None)
        index_2 = [slice(1, -1)] * N
        index_2[M] = slice(0, -2)
        difference = arr[tuple(index_1)] - arr[tuple(index_2)]
        grad = tf.concat([grad, difference], axis=-1)
    return grad


def divergence(arr):
    """
    :param arr:
    (size_1, size_2, ..., size_N, N)
    :return:
    (size_1, size_2, ..., size_N, 1)
    """
    N = len(tf.shape(arr)[:-1])
    assert tf.shape(arr)[-1] == N
    div = tf.zeros_like(arr)[..., :1]
    arr = periodic_padding(arr)
    for M in range(N):
        index_1 = [slice(1, -1)] * (N + 1)
        index_1[M] = slice(2, None)
        index_1[-1] = slice(M, M + 1)
        index_2 = [slice(1, -1)] * (N + 1)
        index_2[M] = slice(0, -2)
        index_2[-1] = slice(M, M + 1)
        difference = arr[tuple(index_1)] - arr[tuple(index_2)]
        div = div + difference
    return div


def evolve(arr, alpha=0.15, factor=1):
    """
    :param
    arr:
    (size_1, size_2, ..., size_N, N+1) the last dimension of last axis supposed to be out of the space
    alpha:
    float
    :return:
    (size_1, size_2, ..., size_N, N+1)
    """
    exchange_field = neighbor_sum(arr) / 2.
    grad = gradient(arr[..., -1:]) * factor
    div = divergence(arr[..., :-1])
    dm_field = tf.concat([-grad, div], axis=-1) * alpha
    return tf.math.l2_normalize(exchange_field + dm_field, axis=-1)


def evolve_no_div(arr, alpha):
    exchange_field = neighbor_sum(arr) / 2.
    grad = -gradient(arr[..., -1:]) * alpha
    div = tf.zeros_like(divergence(arr[..., :-1]))
    dm_field = tf.concat([-grad, div], axis=-1) * alpha
    return tf.math.l2_normalize(exchange_field + dm_field, axis=-1)


def slice_index(N):
    """
    :param N: int
    :return: ((N!, N+1, N), (N!,))
    """

    slices_index = np.zeros((factorial(N), N + 1, N), dtype=int)
    possible_direction = np.eye(N, dtype=int)
    possible_route = np.array(list(permutations(possible_direction)))
    for i in range(N):
        slices_index[:, i + 1] = (slices_index[:, i] + possible_route[:, i])

    sign = np.linalg.det(slices_index[:, 1:]).astype(np.float32)
    return slices_index, sign


def get_solid_angle(arr, pad=True, use_one_simplex=False):
    N = len(tf.shape(arr)[:-1])
    assert tf.shape(arr)[-1] == N + 1
    assert 2 <= N
    slices_index, signs = slice_index(N)
    if use_one_simplex:
        slices_index, signs = slices_index[:1], signs[:1]

    if pad:
        padded = periodic_padding(arr, 'uni')
    else:
        padded = arr
    solid_angle = tf.zeros_like(padded)[[slice(-1) for _ in range(N)]][..., 0]
    for sequence_index, sign in zip(slices_index, signs):
        spins = tf.zeros_like(padded)[[slice(-1) for _ in range(N)]][..., tf.newaxis][..., :0]
        for index in sequence_index:
            index = list(map(lambda x: slice(0, -1) if x == 0 else slice(1, None), index))
            spins = tf.concat([
                spins,
                padded[index][..., tf.newaxis]
            ], axis=-1)
        solid_angle = solid_angle + tf.linalg.det(spins) * sign
    constant = 2 * np.pi ** (int(N + 1) / 2.) / gamma(int(N + 1) / 2.)
    solid_angle = tf.reduce_sum(solid_angle) / constant
    if not use_one_simplex:
        solid_angle = solid_angle / factorial(N)
    return solid_angle


def slice_arr(arr, size=5):
    shape = arr.shape
    assert shape[0] % size == 0
    assert len(shape) == shape[-1]
    N = shape[-1] - 1
    arr = periodic_padding(arr, 'uni')
    batch = tf.zeros([0] + [size + 1 for _ in range(N)] + [N + 1])
    for indices in np.ndindex(*[shape[0] // size for _ in range(N)]):
        batch = tf.concat([batch, arr[tuple(slice(size * i, size * (i + 1) + 1) for i in indices)][tf.newaxis]], axis=0)
    return batch
def get_solid_angle_fine(arr, size=5, ratio=10, use_one_simplex=True):
    """up-scales the input array to get fine topological number"""
    shape = arr.shape
    # assert ratio != 1
    assert shape[0] % size == 0
    assert len(shape) == shape[-1]
    N = shape[-1] - 1
    arr = slice_arr(arr, size=size)
    solid_angle = 0.
    for i, sub_arr in enumerate(arr):
        # sub_arr = zoom(sub_arr, [ratio for _ in range(N)] + [1], order=1)
        # sub_arr = tf.math.l2_normalize(sub_arr, axis=-1)

        for axis in range(N):
            perm = list(range(N + 1))
            perm[0], perm[axis] = perm[axis], perm[0]
            sub_arr = tf.transpose(sub_arr, perm=perm)
            sub_arr = tf.reshape(sub_arr, [size + 1, -1, sub_arr.shape[-1]])
            sub_arr = tf.image.resize(sub_arr, [(size + 1) * ratio, sub_arr.shape[1]])
            sub_arr = tf.reshape(sub_arr, [(size + 1) * ratio for _ in range(axis + 1)] + [size + 1 for _ in
                                                                                           range(N - axis - 1)] + [-1])
            # sub_arr = tf.image.resize(sub_arr, [size*ratio + 1, sub_arr.shape[1]])
            # sub_arr = tf.reshape(sub_arr, [size*ratio + 1 for _ in range(axis+1)] + [size+1 for _ in range(N-axis-1)] + [-1])
            sub_arr = tf.transpose(sub_arr, perm=perm)

        sub_arr = tf.math.l2_normalize(sub_arr, axis=-1)

        solid_angle = solid_angle + get_solid_angle(tf.cast(sub_arr, tf.float32), pad=False, use_one_simplex=use_one_simplex)

        print("\r {} / {}".format(i + 1, len(arr)), end="")
    return solid_angle

def get_B4_mask(size: int = 30):
    space_limit = 1.
    y_space = tf.transpose(tf.meshgrid(*[tf.linspace(-space_limit, space_limit, size) for i in range(4)]), [1, 2, 3, 4, 0])
    r = tf.norm(y_space, axis=-1)
    mask = tf.greater(0.5, r)
    return mask
def get_S3_mask(size: int = 30):
    space_limit = 1.5
    maximum_distance = space_limit * 4. / float(size - 1)
    y_space = tf.transpose(tf.meshgrid(*[tf.linspace(-space_limit, space_limit, size) for i in range(4)]), [1, 2, 3, 4, 0])
    r = tf.norm(y_space, axis=-1)
    mask = tf.greater(maximum_distance, tf.abs(r - 1.))
    return mask
def get_real_projective_plane_mask_4d(size: int = 30) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Implementation of real projective plane with embeded on 4D square lattice.
    """
    space_limit = 1.5
    maximum_distance = space_limit * 4. / float(size - 1)
    y_space = tf.transpose(tf.meshgrid(*[tf.linspace(-space_limit, space_limit, size) for i in range(4)]), [1, 2, 3, 4, 0])

    def embed_projective_plane(x):
        theta = x[..., 0]
        phi = x[..., 1]
        x1 = tf.sin(theta) * tf.cos(phi)
        x2 = tf.sin(theta) * tf.sin(phi)
        x3 = tf.cos(theta)
        y1 = x1 * x2
        y2 = x1 * x3
        y3 = x2 ** 2 - x3 ** 2
        y4 = x2 * x3 * 2.
        return tf.stack([y1, y2, y3, y4], axis=-1)

    x_init = tf.zeros((size, size, size, size, 2))
    x_variable = tf.Variable(x_init)
    optimizer = tf.optimizers.Adam()
    for i in range(10000):
        with tf.GradientTape() as t:
            y = embed_projective_plane(x_variable)
            differnece = tf.norm(y - y_space, axis=-1)
            tf.print("\r", i, "/ 10000   Averaged Distance: ", tf.reduce_mean(differnece).numpy(), end="")
        gradient = t.gradient(differnece, x_variable)
        optimizer.apply_gradients([(gradient, x_variable)])
    print("")
    x_final = x_variable.read_value()
    distance_from_the_embedded_surface = tf.norm(embed_projective_plane(x_final) - y_space, axis=-1)
    mask = tf.cast(tf.greater(maximum_distance, distance_from_the_embedded_surface), tf.float32)
    return
def get_klein_bottle_mask_4d(size: int = 30) -> Tuple[tf.Tensor, tf.Tensor]:
    space_limit = 1.5
    maximum_distance = space_limit * 4. / float(size - 1)
    y_space = tf.transpose(tf.meshgrid(*[tf.linspace(-space_limit, space_limit, size) for i in range(4)]), [1, 2, 3, 4, 0])
    def embed_klein_bottle(x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = (1. + tf.cos(x1)) * tf.cos(x2) / 2.
        y2 = (1. + tf.cos(x1)) * tf.sin(x2) / 2.
        y3 = tf.sin(x1) * tf.cos(x2 / 2.)
        y4 = tf.sin(x1) * tf.sin(x2 / 2.)
        return tf.stack([y1, y2, y3, y4], axis=-1)
    x_init = tf.zeros((size, size, size, size, 2))
    x_variable = tf.Variable(x_init)
    optimizer = tf.optimizers.Adam()
    for i in range(10000):
        with tf.GradientTape() as t:
            y = embed_klein_bottle(x_variable)
            differnece = tf.norm(y - y_space, axis=-1)
            tf.print("\r", i, "/ 10000   Averaged Distance: ", tf.reduce_mean(differnece).numpy(), end="")
        gradient = t.gradient(differnece, x_variable)
        optimizer.apply_gradients([(gradient, x_variable)])
    print("")
    x_final = x_variable.read_value()
    distance_from_the_embedded_surface = tf.norm(embed_klein_bottle(x_final) - y_space, axis=-1)
    mask = tf.cast(tf.greater(maximum_distance, distance_from_the_embedded_surface), tf.float32)
    return mask
def get_object_from_mask(mask):
    N = len(tf.shape(mask))
    last_dim = tf.cast(mask, tf.float32) * 2. - 1.
    full_mapping = tf.stack([tf.zeros_like(last_dim) for i in range(N)] + [last_dim], axis=-1)
    return full_mapping

    
if __name__ == "__main__":
    print("Constructing Object")
    mask = get_B4_mask() # B4 implemented in 4D square lattice
    # mask = get_S3_mask() # S3 Hyperspehere implemented in 4D square lattice
    # mask = get_real_projective_plane_mask_4d() # Real Projective Plane implemented in 4D square lattice
    # mask = get_klein_bottle_mask_4d() # Klein Bouttle implemented in 4D square lattice

    object = get_object_from_mask(mask)


    print("Constructing S4 Mapping")
    for _ in range(20):
        object = evolve(object, 0.1)
    object = np.array(object)

    print("Computing Solid angle")
    solid_angle = get_solid_angle(object, use_one_simplex=False)
    print("n=", solid_angle.numpy(), "(Regularly computed topological number. This may provide a poor result.)")
    print("")
    print("Up-scaling technique started")
    solid_angle_fine = get_solid_angle_fine(object, size=5, ratio=10, use_one_simplex=True)
    print("")
    print("n=", solid_angle_fine.numpy(), "(Topological number computed from up-scaled spin configuration. It may provide a better result)")
