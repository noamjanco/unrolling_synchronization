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


def generate_training_data_z_over_2(N: int, Lambda: float, R: int) -> [np.array, np.array]:
    """
    Generate training data for Z/2, the training data consists of R trials
    :param N: Number of observations in each trial, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param R: Number of trials
    :return: x_total - set of vector we wish to estimate, x in {+-1}^N,
             Y_total - R x N x N relative measurements matrix
    """
    Y_total = []
    x_total = []
    for r in range(R):
        x, Y = generate_data_z_over_2(N, Lambda)
        x_total.append(x)
        Y_total.append(Y)

    x_total = np.asarray(x_total)
    Y_total = np.asarray(Y_total)

    print('Finished generating training data')
    return x_total, Y_total


def generate_data_z_over_2_bsc(N: int, Lambda: float) -> [np.array, np.array]:
    """
    Generate target vector x and measurements Y, in Z/2 scenario.

    :param N: Number of observations, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :return: x - vector we wish to estimate, x in {+-1}^N,
             Y - N x N relative measurements matrix
    """
    # noinspection PyTypeChecker
    x = np.expand_dims(2*(np.random.randint(0,2,N))-1,axis=-1)
    p_err = min(1/(2*Lambda),0.5)
    a = np.random.choice([1, -1], (N, N), replace=True, p=[1 - p_err, p_err])
    W = np.tril(a) + np.tril(a, -1).T
    Y = Lambda / N * x @ x.T * W
    return x, Y


def generate_training_data_z_over_2_bsc(N: int, Lambda: float, R: int) -> [np.array, np.array]:
    """
    Generate training data for Z/2, the training data consists of R trials
    :param N: Number of observations in each trial, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param R: Number of trials
    :return: x_total - set of vector we wish to estimate, x in {+-1}^N,
             Y_total - R x N x N relative measurements matrix
    """
    Y_total = []
    x_total = []
    for r in range(R):
        x, Y = generate_data_z_over_2_bsc(N, Lambda)
        x_total.append(x)
        Y_total.append(Y)

    x_total = np.asarray(x_total)
    Y_total = np.asarray(Y_total)

    print('Finished generating training data')
    return x_total, Y_total


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


def generate_data_antipodal(N: int, Lambda: float, L: int) -> [np.array, np.array, np.array, np.array]:
    """
    Generate target flip vector s and relative measurements H in U(1) antipodal signal model.

    :param N: Number of observations, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param L: Number of possible rotations - corresponds to the length of a vector in MRA experiments.
    :return: s - vector we wish to estimate, x in {+-1}^N,
             H - N x N relative measurements matrix
             y - N x L measurements matrix
             x - Unknown signal
    """
    x = np.random.randn(L)
    y = np.zeros((L, N))
    n = np.zeros((L, N))
    s = np.zeros((N,), dtype=int)
    sigma = 1/Lambda

    for i in range(N):
        s[i] = 2*int(np.random.randint(0, 2)) - 1
        n[:, i] = sigma * np.random.randn(L)
        y[:, i] = x * s[i] + n[:, i]

    H = np.zeros((N, N),dtype=np.float32)
    for n in range(N):
        for m in range(N):
            H[n, m] = Lambda / N * y[:,n] @ y[:,m].T

    return np.expand_dims(s,axis=-1).astype(np.float32), H, y, x


def generate_training_data_antipodal(N: int, Lambda: float, R: int, L: int) -> [np.array, np.array]:
    """
    Generate training data for Z/2, the training data consists of R trials
    :param N: Number of observations in each trial, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param R: Number of trials
    :return: x_total - set of vector we wish to estimate, x in {+-1}^N,
             Y_total - R x N x N relative measurements matrix
    """
    Y_total = []
    x_total = []
    for r in range(R):
        x, Y, _, _ = generate_data_antipodal(N, Lambda, L)
        x_total.append(x)
        Y_total.append(Y)

    x_total = np.asarray(x_total)
    Y_total = np.asarray(Y_total)

    print('Finished generating training data')
    return x_total, Y_total

def generate_training_data_antipodal_reconstruction(N: int, Lambda: float, R: int, L: int) -> [np.array, np.array, np.array, np.array]:
    """
    Generate training data for Z/2, the training data consists of R trials
    :param N: Number of observations in each trial, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param R: Number of trials
    :return: s_total - set of rotation vector we wish to estimate, s in {+-1}^N,
             Y_total - R x N x N relative measurements matrix
             y_total - measured noisy and rotated signal, R X N X L
             x_total - Ground truth signal , R X L
    """
    Y_total = []
    s_total = []
    y_total = []
    x_total = []
    for r in range(R):
        s, Y, y, x = generate_data_antipodal(N, Lambda, L)
        s_total.append(s)
        Y_total.append(Y)
        y_total.append(y)
        x_total.append(x)

    s_total = np.asarray(s_total)
    Y_total = np.asarray(Y_total)
    y_total = np.asarray(y_total)
    x_total = np.asarray(x_total)

    print('Finished generating training data')
    return s_total, Y_total, y_total, x_total