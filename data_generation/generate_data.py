import numpy as np
import typing

def generate_data_z_over_2(N: int, Lambda: float) -> [np.array, np.array]:
    """
    Generate target vector x and measurements Y, in Z/2 scenario.

    :param N: Number of observations, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :return: x - vector we wish to estimate, x in {+-1}^N,
             Y - N x N relative measurements matrix
    """
    # noinspection PyTypeChecker
    x = np.expand_dims(2*(np.random.randint(0,2,N))-1,axis=-1)
    a = np.random.randn(N, N)
    W = np.tril(a) + np.tril(a, -1).T
    Y = Lambda / N * x@x.T + 1/np.sqrt(N) * W
    return x, Y


def generate_data_gaussian(N: int, Lambda: float, L: int) -> [np.array, np.array, np.array]:
    """
    Generate target rotations vector x and relative rotations Y, and discrete matching shifts in U(1) gaussian scenario.

    :param N: Number of observations, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param L: Number of possible rotations - corresponds to the length of a vector in MRA experiments.
    :return: x - vector we wish to estimate, x in {C}^N,
             Y - N x N relative measurements matrix
             s - ground truth discrete shifts in [0,..,L-1]
    """
    s = np.random.randint(0,L,N)
    # s = np.random.rand(N)*L
    angles = 2 * np.pi * s / L
    x = np.expand_dims(np.exp(1j * angles),axis=-1)
    a = 1/np.sqrt(2)*(np.random.randn(N, N) + 1j*np.random.randn(N, N))
    W = np.tril(a) + np.conj(np.tril(a, -1).T)
    Y = Lambda / N * x@np.conj(x.T) + 1/np.sqrt(N) * W
    return x, Y, s
