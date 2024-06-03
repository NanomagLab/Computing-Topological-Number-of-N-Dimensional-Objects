"""
This file contains the implementation for processing data of arbitrary dimensions.
The code was developed during our research process and may be more complex and harder to understand.
It is recommended for advanced users who need to work with data of arbitrary dimensions.
"""
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, gamma
from scipy.ndimage import zoom
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

        # spins = tf.stack([
        #     padded[list(map(lambda x: slice(0, -1) if x == 0 else slice(1, None), index))] for index in sequence_index
        # ], axis=-1)
        solid_angle = solid_angle + tf.linalg.det(spins) * sign
    constant = 2 * np.pi ** (int(N + 1) / 2.) / gamma(int(N + 1) / 2.)
    return tf.reduce_sum(solid_angle) / constant / factorial(N)


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
    print("")
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


if __name__ == "__main__":
    # List available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPUs available:", gpus)

    # Enable memory growth for each GPU
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")
        except RuntimeError as e:
            # Memory growth must be set before any TensorFlow operations
            print(e)
    size = 30
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    w = np.linspace(-1, 1, size)
    xx, yy, zz, ww = np.meshgrid(x, y, z, w)
    # xxx,yyy=np.meshgrid(x,y)
    # yyyy,zzzz=np.meshgrid(y,z)
    # 3D 토러스 생성
    # radius = 0.25
    # c=0.6
    # spin_mask = (c-np.sqrt(xx**2 + yy**2))**2 + zz**2 <= radius**2

    # 3D 구 생성
    radius = 0.75
    spin_mask = xx ** 2 + yy ** 2 + zz ** 2 + ww ** 2 <= radius ** 2
    radius2 = 0.5
    spin_mask2 = xx ** 2 + yy ** 2 + zz ** 2 >= radius2 ** 2
    spin_mask = spin_mask * spin_mask2

    # # 3d 손잡이
    # def get_3d_grid(sizex, sizey, sizez, initx = 0, inity = 0, initz = 0):
    #     for x in range(initx, sizex):
    #         for y in range(inity, sizey):
    #             for z in range(initz, sizez):
    #                 yield (x, y, z)
    #
    spin_data = np.zeros((size, size, size, size, 5))
    spin_data = spin_data - 1
    spin_data[spin_mask, :] = +1
    # spin_data[spin_mask3,0:30,:] = +1
    #
    # spin_data[:,:,0:15,: ]=-1
    # spin_data[:,:,75:90,: ]=-1
    # for (x, y, z) in get_3d_grid(size, size, size):
    #     if (z > 25 and z < 65) and (y > 65 and y < 83) and (x > 40 and x < 50):
    #         spin_data[x][y][z] = [0, 0, 0, 1]
    #     if (z > 35 and z < 55) and (y > 65 and y < 75) and (x > 40 and x < 50):
    #         spin_data[x][y][z] = [0, 0, 0, -1]

    spin_data[:, :, :, : 0:4] = 0.

    sphere = spin_data.astype(np.float32)
    for _ in range(20):
        sphere = evolve(sphere, 0.1)
    sphere = np.array(sphere)
    plt.imshow(sphere[:, :, size // 2, size // 2, -1])
    plt.show()
    print("init_cond complete")
    solid_angle = get_solid_angle(sphere, use_one_simplex=False)
    print("n=", solid_angle.numpy(), "(Regularly computed topological number. It might show a poor result.)")

    print("Up-scaling technique started")
    solid_angle_fine = get_solid_angle_fine(sphere, size=6, ratio=10, use_one_simplex=True)
    print("")
    print("n=", solid_angle_fine.numpy(), "(Topological number computed from Up-scaled spin configuration. It might show a better result)")

