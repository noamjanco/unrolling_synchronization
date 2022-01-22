import numpy as np
import tensorflow as tf

def rel_error_z_over_2(y_true: np.array, y_pred: np.array) -> float:
    """
    Computes the relative error between two vectors in Z/2. Vectors might be the same up to the orbit.
    In Z/2, the error is the minimum across the vector orbit which included both signs.
    :param y_true: True vector
    :param y_pred: Predicted vector
    :return: a float indicating the error up to orbit between the true and predicted vectors
    """
    y_true = np.expand_dims(y_true,axis=0)
    y_pred = np.expand_dims(y_pred,axis=0)
    err = 1 / 4 * tf.reduce_mean(tf.math.minimum(tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=1),
                                                 tf.reduce_mean(tf.pow(y_true + y_pred, 2), axis=1)))
    return err


def rel_error_u_1(y_true: np.array, y_pred: np.array) -> float:
    """
    Computes the relative error between two vectors in U(1). Vectors might be the same up to phase shift.
    :param y_true: True vector
    :param y_pred: Predicted vector
    :return: a float indicating the error up to orbit between the true and predicted vectors
    """
    err = 1 - np.abs(np.vdot(y_true,y_pred))/y_true.shape[0]
    return err


def normalize(z):
    z_normalized = z / np.abs(z)
    z_normalized[z_normalized == 0] = 0
    return z_normalized

def align_in_fourier(y: np.ndarray, z: np.ndarray, L: int):
    """
    This function de-rotates the measurements Matrix y (given in fourier), by the estimated phase vector z,
    and then averages over the aligned measurements.
    :param y: Measurements matrix in fourier domain - L X N
    :param z: Estimated phase vector in {C}^N
    :param L: Length of the measurements
    :return: Mean of aligned MRA measurements, in fourier.
    """
    #todo: remove L from inputs
    s_est = (np.angle(-z) / (2 * np.pi) * L)[:, 0]

    e_r = tf.math.cos(2 * np.pi * tf.expand_dims(tf.range(L, dtype=tf.float32), axis=-1) @ tf.expand_dims(
        tf.cast(s_est, dtype=tf.float32), axis=0) / L)
    e_i = tf.math.sin(2 * np.pi * tf.expand_dims(tf.range(L, dtype=tf.float32), axis=-1) @ tf.expand_dims(
        tf.cast(s_est, dtype=tf.float32), axis=0) / L)
    y_a_r = e_r * y.real - e_i * y.imag
    y_a_i = e_r * y.imag + e_i * y.real
    y_a_r_mean = tf.reduce_mean(y_a_r, axis=-1)
    y_a_i_mean = tf.reduce_mean(y_a_i, axis=-1)

    x_est = tf.concat([y_a_r_mean, y_a_i_mean], axis=-1)
    return x_est