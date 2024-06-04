"""
This file contains the implementation for 2D data processing.
The code is written to be user-friendly and easy to understand,
suitable for those who need to work with 2D data.
"""
import tensorflow as tf


def periodic_padding(arr):
    """
    :param arr: array of shape (height, width, ...)
    :return: array of shape (height + 2, width + 2, ...)
    """
    padded = tf.concat([arr[-1:], arr, arr[:1]], axis=0)
    padded = tf.concat([padded[:, -1:], padded, padded[:, :1]], axis=1)
    return padded


def gradient(arr):
    """
    :param arr: array of shape (height, width, 1)
    :return: array of shape (height, width, 2)
    """
    padded = periodic_padding(arr)
    grad_x = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.
    grad_y = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.
    return tf.concat([grad_x, grad_y], axis=-1)


def divergrence(arr):
    """
    :param arr: array of shape (height, width, 2)
    :return:  array of shape (height, width, 1)
    """
    padded = periodic_padding(arr)
    dx = (padded[2:, 1:-1, 0] - padded[:-2, 1:-1, 0]) / 2.
    dy = (padded[1:-1, 2:, 1] - padded[1:-1, :-2, 1]) / 2.
    return tf.expand_dims(dx + dy, axis=-1)


def sum_nearist_neighbor(arr):
    """
    :param arr: array of shape (height, width, ...)
    :return:  array of shape (height, width, ...)
    """
    padded = periodic_padding(arr)
    return padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]


def evolve(arr, alpha=0.15):
    """
    :param arr: array of shape (height, width, 3)
    :return:  array of shape (height, width, 3)
    """
    exchange_field = sum_nearist_neighbor(arr)
    grad = gradient(arr[..., -1:])
    div = divergrence(arr[..., :-1])
    dm_field = tf.concat([-grad, div], axis=-1) * alpha
    return tf.math.l2_normalize(exchange_field + dm_field, axis=-1)


def gray_to_spin_2d(arr, alpha=0.1, steps=100):
    """
    :param arr: array of shape (height, width, 1)
    :param alpha: float
    :param steps: int
    :return:  array of shape (height, width, 3)
    """
    arr = tf.concat([tf.zeros_like(arr), tf.zeros_like(arr), arr], axis=-1)
    for i in range(steps):
        arr = evolve(arr, alpha=alpha)
    return arr


def get_solid_angle_density_2d(arr):
    """
    :param arr: array of shape (height, width, 3)
    :return:  array of shape (height, width)
    """
    solid_angle = tf.linalg.det(tf.stack([arr[:-1, :-1], arr[1:, :-1], arr[1:, 1:]], axis=-1))
    solid_angle = solid_angle - tf.linalg.det(tf.stack([arr[:-1, :-1], arr[:-1, 1:], arr[1:, 1:]], axis=-1))
    return solid_angle / 2.
