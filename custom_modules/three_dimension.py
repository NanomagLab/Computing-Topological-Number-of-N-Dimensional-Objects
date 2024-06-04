"""
This file contains the implementation for 3D data processing.
The code is written to be user-friendly and easy to understand,
suitable for those who need to work with 3D data.
"""
import tensorflow as tf


def periodic_padding(arr):
    """
    :param arr: array of shape (x, y, z, ...)
    :return: array of shape (x + 2, y + 2, z + 2, ...)
    """
    padded = tf.concat([arr[-1:], arr, arr[:1]], axis=0)
    padded = tf.concat([padded[:, -1:], padded, padded[:, :1]], axis=1)
    padded = tf.concat([padded[:, :, -1:], padded, padded[:, :, :1]], axis=2)
    return padded


def gradient(arr):
    """
    :param arr: array of shape (x, y, z, 1)
    :return: array of shape (x, y, z, 3)
    """
    padded = periodic_padding(arr)
    grad_x = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / 2.
    grad_y = (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / 2.
    grad_z = (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / 2.
    return tf.concat([grad_x, grad_y, grad_z], axis=-1)


def divergrence(arr):
    """
    :param arr: array of shape (x, y, z, 3)
    :return:  array of shape (x, y, z, 1)
    """
    padded = periodic_padding(arr)
    dx = (padded[2:, 1:-1, 1:-1, 0] - padded[:-2, 1:-1, 1:-1, 0]) / 2.
    dy = (padded[1:-1, 2:, 1:-1, 1] - padded[1:-1, :-2, 1:-1, 1]) / 2.
    dz = (padded[1:-1, 1:-1, 2:, 2] - padded[1:-1, 1:-1, :-2, 1]) / 2.
    return tf.expand_dims(dx + dy + dz, axis=-1)


def sum_nearist_neighbor(arr):
    """
    :param arr: array of shape (x, y, z, ...)
    :return:  array of shape (x, y, z, ...)
    """
    padded = periodic_padding(arr)
    return padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1] +\
           padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1] +\
           padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]


def evolve(arr, alpha=0.15):
    """
    :param arr: array of shape (x, y, z, 4)
    :return:  array of shape (x, y, z, 4)
    """
    exchange_field = sum_nearist_neighbor(arr) / 2.
    grad = gradient(arr[..., -1:])
    div = divergrence(arr[..., :-1])
    dm_field = tf.concat([-grad, div], axis=-1) * alpha
    return tf.math.l2_normalize(exchange_field + dm_field, axis=-1)


def gray_to_spin_3d(arr, alpha=0.1, steps=100):
    """
    :param arr: array of shape (x, y, z, 1)
    :param alpha: float
    :param steps: int
    :return:  array of shape (x, y, z, 4)
    """
    arr = tf.concat([tf.zeros_like(arr), tf.zeros_like(arr), tf.zeros_like(arr), arr], axis=-1)
    for i in range(steps):
        arr=evolve(arr, alpha=alpha)
    return arr


def get_solid_angle_density_3d(arr):
    """
    :param arr: array of shape (x, y, z, 4)
    :return:  array of shape (x, y, z)
    """
    solid_angle = -tf.linalg.det(tf.stack([
        arr[:-1, :-1, :-1], arr[1:, :-1, :-1], arr[1:, 1:, :-1], arr[1:, 1:, 1:]
    ], axis=-1))
    solid_angle = solid_angle + tf.linalg.det(tf.stack([
        arr[:-1, :-1, :-1], arr[1:, :-1, :-1], arr[1:, :-1, 1:], arr[1:, 1:, 1:]
    ], axis=-1))
    solid_angle = solid_angle + tf.linalg.det(tf.stack([
        arr[:-1, :-1, :-1], arr[:-1, 1:, :-1], arr[1:, 1:, :-1], arr[1:, 1:, 1:]
    ], axis=-1))
    solid_angle = solid_angle - tf.linalg.det(tf.stack([
        arr[:-1, :-1, :-1], arr[:-1, 1:, :-1], arr[:-1, 1:, 1:], arr[1:, 1:, 1:]
    ], axis=-1))
    solid_angle = solid_angle - tf.linalg.det(tf.stack([
        arr[:-1, :-1, :-1], arr[:-1, :-1, 1:], arr[1:, :-1, 1:], arr[1:, 1:, 1:]
    ], axis=-1))
    solid_angle = solid_angle + tf.linalg.det(tf.stack([
        arr[:-1, :-1, :-1], arr[:-1, :-1, 1:], arr[:-1, 1:, 1:], arr[1:, 1:, 1:]
    ], axis=-1))
    return solid_angle / 6.
