import numpy as np


def generate_data_1d_mra(N: int, Lambda: float, L: int) -> [np.array, np.array, np.array, np.array, np.array]:
    """
    Generate target rotations vector x and relative rotations Y, and discrete matching shifts in U(1) gaussian scenario.

    :param N: Number of observations, also the length of the vector x
    :param Lambda: Signal-to-noise ratio (SNR) parameter
    :param L: Number of possible rotations - corresponds to the length of a vector in MRA experiments.
    :return: z - rotation vector of each measurement (correspond to the cyclic shift of each measurement) in {C}^N
             Y - N x N relative measurements matrix
             s - ground truth discrete shifts in [0,..,L-1]
             y_fft - fft of the noisy measurements vectors
             x_fft - fft of vector we wish to estimate, x in {C}^L,
    """
    x = np.random.randn(L)
    y = np.zeros((L, N))
    n = np.zeros((L, N))
    s = np.zeros((N,), dtype=int)
    sigma = 1/Lambda

    for i in range(N):
        s[i] = int(np.random.randint(0, L))
        n[:, i] = sigma * np.random.randn(L)
        y[:, i] = np.roll(x, s[i]) + n[:, i]

    Y = np.fft.fft(y, axis=0)
    rho = np.zeros((N, N))
    for n in range(N):
        for m in range(n):
            R_nm = np.fft.ifft(Y[:, n] * Y[:, m].conj())
            k = np.argmax(R_nm)
            rho[n, m] = k
    for n in range(N):
        for m in range(n, N):
            rho[n, m] = (L - rho[m, n]) % L

    H = Lambda / N *np.exp(1j * 2 * np.pi * rho / L)
    z = np.expand_dims(np.exp(1j*s * 2*np.pi / L),axis=-1)
    y_fft = np.fft.fft(y,axis=0)
    x_fft = np.fft.fft(x,axis=0)

    return z, H, s, y_fft, x_fft


def generate_training_data_1d_mra(N: int, Lambda: float, R: int, L: int):
    z_total = []
    Y_total = []
    s_total = []
    y_fft_total = []
    x_fft_total = []
    for r in range(R):
        z, Y, s, y_fft, x_fft = generate_data_1d_mra(N, Lambda, L)
        z_total.append(z)
        Y_total.append(Y)
        s_total.append(s)
        y_fft_total.append(y_fft)
        x_fft_total.append(x_fft)

    z_total = np.asarray(z_total)
    Y_total = np.asarray(Y_total)
    s_total = np.asarray(s_total)
    y_fft_total = np.asarray(y_fft_total)
    x_fft_total = np.asarray(x_fft_total)

    print('Finished generating training data')
    return s_total, Y_total, z_total, y_fft_total, x_fft_total