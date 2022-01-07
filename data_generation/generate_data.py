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
