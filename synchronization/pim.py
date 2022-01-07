import numpy as np
import typing


def pim_z_over_2(Y: np.array, v_init: np.array, max_iterations: int=1000, tol: float=1e-5) -> [np.array, int]:
    """
    Power iteration method synchronization for Z/2 case.
    :param Y: N x N Measurement matrix, where N is the length of v_init
    :param v_init: Initial guess vector, N x 1
    :param max_iterations: maximum number of iterations
    :param tol: tolerance for stopping condition
    :return: Estimated vector v, and number of iterations
    """
    v = v_init
    i = 0
    while i < max_iterations:
        v_prev = v
        v = Y @ v_prev
        v = v / np.sqrt(np.sum(v ** 2))
        if np.sum((v - v_prev) ** 2) < tol:
            break
        i += 1
    return np.sign(v), i
