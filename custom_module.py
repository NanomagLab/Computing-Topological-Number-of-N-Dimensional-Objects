import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def periodic_padding_2d(arr, pad=1):
    """
    :param arr: array of shape (height, width, ...)
    :param pad: int
    :return: array of shape (height + pad*2, width + pad*2, ...)
    """
    padded = tf.concat([arr[-1:], arr, arr[:1]], axis=0)
    padded = tf.concat([padded[:, -1:], padded, padded[:, :1]], axis=1)
    return padded


def gradient_2d(arr):
    """
    :param arr: array of shape (height, width, 1)
    :return: array of shape (height, width, 2)
    """
    padded = periodic_padding_2d(arr)
    grad_x = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.
    grad_y = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.
    return tf.concat([grad_x, grad_y], axis=-1)


def divergence_2d(arr):
    """
    :param arr: array of shape (height, width, 2)
    :return:  array of shape (height, width, 1)
    """
    padded = periodic_padding_2d(arr)
    dx = (padded[2:, 1:-1, 0] - padded[:-2, 1:-1, 0]) / 2.
    dy = (padded[1:-1, 2:, 1] - padded[1:-1, :-2, 1]) / 2.
    return tf.expand_dims(dx + dy, axis=-1)


def sum_nearist_neighbor_2d(arr):
    """
    :param arr: array of shape (height, width, ...)
    :return:  array of shape (height, width, ...)
    """
    padded = periodic_padding_2d(arr)
    return padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]


def evolve_2d(arr, alpha=0.15):
    """
    :param arr: array of shape (height, width, 3)
    :return:  array of shape (height, width, 3)
    """
    exchange_field = sum_nearist_neighbor_2d(arr) / 2.
    grad = gradient_2d(arr[..., -1:])
    div = divergence_2d(arr[..., :-1])
    dm_field = tf.concat([-grad, div], axis=-1) * alpha
    return tf.math.l2_normalize(exchange_field + dm_field, axis=-1)


# def build_disk(size=50, radius=0.85):
#     meshgrid = np.meshgrid(*[np.linspace(-1., 1., size, dtype=np.float32) for _ in range(2)])
#     mask = np.sum(np.square(meshgrid), axis=0) <= radius ** 2.
#     arr = -np.ones((size, size, 3), dtype=np.float32)
#     arr[mask] = 1.
#     arr[..., 0:-1] = 0.
#     return arr
#
#
# arr = build_disk(size=50)
#
#
# for _ in range(20):
#     arr = evolve_2d(arr, 0.2)


def gray_to_spin_2d(arr, alpha=0.2, steps=20):
    """
    :param arr: array of shape (height, width, 3)
    :param alpha: float
    :param steps: int
    :return:  array of shape (height, width, 3)
    """
    for i in range(steps):
        evolve_2d(arr, alpha=alpha)
    return arr


def get_solid_angle_density_2d(arr):
    """
    :param arr: array of shape (height, width, 3)
    :return:  array of shape (height, width)
    """
    solid_angle = tf.linalg.det(tf.stack([arr[:-1, :-1], arr[1:, :-1], arr[:-1, 1:]], axis=-1))
    solid_angle = solid_angle + tf.linalg.det(tf.stack([arr[1:, :-1], arr[1:, 1:], arr[:-1, 1:]], axis=-1))
    return solid_angle


# def get_solid_angle_2d_fine(arr, patch_size=5, ratio=10):
#     def get_patches(arr, patch_size=5, ratio=10):
#         arr = periodic_padding_2d(arr)[1:, 1:]
#         patches = tf.zeros((0, ))
#
#
#     assert arr.shape[0] == arr.shape[1]
#     assert arr.shape[0] % patch_size == 0
#     solid_angle = 0.
#     padded = periodic_padding_2d(arr)[1:, 1:]
#     for i in range(arr.shape[0] // patch_size):
#         for j in range(arr.shape[1] // patch_size)
#             sub_arr = arr[i * patch_size:i * patch_size + patch_size + 1, j * patch_size:j * patch_size + patch_size + 1]
#             sub_arr = tf.image.resize(sub_arr, ((size+1) * ratio, sub))
