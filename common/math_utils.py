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


def squared_correlation(X, X_hat) -> float:
    N = X.shape[0]/3
    return np.linalg.norm(X.T @ X_hat) / (N * np.sqrt(3))

def rel_error_so3(X, X_hat) -> float:
    return 1 - squared_correlation(X, X_hat)




def project_to_orthogonal_matrix(M: np.array, flip=False):
    """
    Projects a 3x3 matrix to an orthogonal matrix.
    This is done by computing USV^T = M
    The orthogonal matrix UV^T is kept.
    return det(UV^T) * UV^T
    :param M: The matrix
    :return: Returns the orthogonal matrix
    """
    u, s, vh = np.linalg.svd(M)
    m_proj = u @ vh

    if flip:
        # todo: think if this necessary in PIM when projecting
        m_proj = np.linalg.det(m_proj) * m_proj
        assert np.isclose(np.linalg.det(m_proj),1)

    return m_proj


def initialize_matrix(N: int) -> np.ndarray:
    z_list = [project_to_orthogonal_matrix(np.random.randn(3,3), flip=True) for _ in range(N)]
    z = np.zeros((3*N, 3))
    for i in range(N):
        z[3*i:3*i+3,:] = z_list[i]
    return z



def project(M: np.ndarray) -> np.ndarray:
    """
    This function projects each 3x3 block of M into the space of orthogonal matrices
    :param M: 3N x 3 Matrix
    :return: a projected matrix, 3N X 3
    """
    N = int(M.shape[0] / 3)
    # M_proj = np.zeros_like(M)
    # M_proj_list = [project_to_orthogonal_matrix(M[3*i:3*i+3,:]) for i in range(N)]
    # for i in range(N):
    #     M_proj[3*i:3*i+3,:] = M_proj_list[i]
    # return M_proj

    M_reshaped = tf.reshape(M, [N, 3, 3])
    norm = tf.sqrt(tf.reduce_sum(tf.pow(M_reshaped,2),axis=[1,2],keepdims=True))
    M_reshaped = M_reshaped / norm

    Q = M_reshaped
    for i in range(10):
        N = tf.matmul(tf.transpose(Q,perm=[0,2,1]), Q)
        P = 1 / 2 * tf.matmul(Q, N)
        Q = 2 * Q + tf.matmul(P, N) - 3 * P

    Q_reshaped = tf.reshape(Q, tf.shape(M)).numpy()

    return Q_reshaped

def project_batch(M: np.ndarray) -> np.ndarray:
    """
    This function projects each 3x3 block of M into the space of orthogonal matrices
    :param M: 3N x 3 Matrix
    :return: a projected matrix, 3N X 3
    """
    BatchSize = M.shape[0]
    N = int(M.shape[1] / 3)
    M_proj = np.zeros_like(M)
    for k in range(BatchSize):
        M_proj_list = [project_to_orthogonal_matrix(M[k,3*i:3*i+3,:]) for i in range(N)]
        for i in range(N):
            M_proj[k,3*i:3*i+3,:] = M_proj_list[i]
    return M_proj
