import numpy as np
import scipy.special
from common.math_utils import rel_error_u_1, normalize


def amp_z_over_2(Y: np.array, v_init: np.array, v_init2: np.array, Lambda: float, max_iterations: int=1000, tol: float=1e-5) -> [np.array, int]:
    """
    Approximate message passing synchronization for Z/2 case.
    :param Y: N x N Measurement matrix, where N is the length of v_init
    :param v_init: Initial guess vector, N x 1
    :param v_init2: 2nd Initial guess vector, N x 1
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param max_iterations: maximum number of iterations
    :param tol: tolerance for stopping condition
    :return: Estimated vector v, and number of iterations
    """
    v = v_init
    v_prev = v_init2

    i = 0
    while i < max_iterations:
        c = Lambda * (Y @ v) - Lambda ** 2 * (1 - np.mean(v ** 2)) * v_prev
        v_prev = v
        v = np.tanh(c)
        if np.sum((v - v_prev) ** 2) < tol:
            break
        i += 1
    return np.sign(v), i


def f(t):
    return scipy.special.i1(2*t)/scipy.special.i0(2*t)


def amp_u_1(Y: np.array, v_init: np.array, v_init2: np.array, Lambda: float, max_iterations: int=1000, tol: float=1e-5) -> [np.array, int]:
    """
    Approximate message passing synchronization for U(1) case.
    :param Y: N x N Measurement matrix, where N is the length of v_init
    :param v_init: Initial guess vector, N x 1
    :param v_init2: 2nd Initial guess vector, N x 1
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param max_iterations: maximum number of iterations
    :param tol: tolerance for stopping condition
    :return: Estimated vector v, and number of iterations
    """
    v = v_init
    v_prev = v_init2

    i = 0
    while i < max_iterations:
        c = Lambda * (Y @ v) - Lambda ** 2 * (1 - np.mean(np.abs(v) ** 2)) * v_prev
        v_prev = v
        v = f(np.abs(c)) * (c / np.abs(c))
        if rel_error_u_1(v, v_prev) < tol:
            break
        i += 1
    return normalize(v), i


