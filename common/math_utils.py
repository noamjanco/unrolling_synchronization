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
