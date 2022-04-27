import numpy as np
import typing

from common.math_utils import normalize, rel_error_u_1, initialize_matrix, project
from typing import List


def ppm_z_over_2(Y: np.array, v_init: np.array, max_iterations: int=1000, tol: float=1e-5) -> [np.array, int]:
    """
    Projected power method synchronization for Z/2 case.
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
        v = np.sign(Y @ v_prev)
        if np.sum((v - v_prev) ** 2) < tol:
            break
        i += 1
    return v, i


def ppm_u_1(Y: np.array, v_init: np.array, max_iterations: int=1000, tol: float=1e-5) -> [np.array, int]:
    """
    Projected power method synchronization for U(1) case.
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
        v = normalize(Y @ v_prev)
        if rel_error_u_1(v, v_prev) < tol:
            break
        i += 1
    return v, i


def ppm_so3(H: np.ndarray, num_iterations: int = 200, tol : float = 1e-3, z_init: np.ndarray = None) -> (List[np.ndarray], int):
    # Extract N from the size of H
    N = int(H.shape[0] / 3)

    # Initialize the matrix z
    if z_init is None:
        z = initialize_matrix(N)
    else:
        z = z_init

    for i in range(num_iterations):
        z_tag = project(H @ z)
        if np.linalg.norm(z_tag-z) < tol:
            break
        z = z_tag

    return z_tag, i