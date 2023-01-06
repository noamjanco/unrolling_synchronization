import numpy as np
import typing
from typing import List
import matplotlib.pyplot as plt

from common.math_utils import squared_correlation, project_to_orthogonal_matrix
from synchronization.pim import pim_so3
from synchronization.ppm import ppm_so3


def apply_j(H: np.ndarray, p : float = 0.5):
    J = np.diag((1,1,-1))
    N = int(H.shape[0] / 3)
    d = {}
    for i in range(N):
        for j in range(i+1,N):
            if np.random.rand() < p:
                d[(i,j)] = 1
                H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = J @ H[3 * i:3 * i + 3, 3 * j:3 * j + 3] @ J
                H[3 * j:3 * j + 3, 3 * i:3 * i + 3] = J @ H[3 * j:3 * j + 3, 3 * i:3 * i + 3] @ J
    return H, d


def generate_data_so3(N: int, Lambda: float) -> [List[np.ndarray], np.ndarray]:

    H = np.zeros((3*N, 3*N))
    R = [project_to_orthogonal_matrix(np.random.randn(3,3), flip=True) for _ in range(N)]
    # W = [project_to_orthogonal_matrix(np.random.randn(3,3), flip=True) for _ in range(N)]

    # for i in range(N):
    #     for j in range(N):
    #         H[3*i:3*i+3,3*j:3*j+3] = R[i] @ R[j].T

    R_mat = np.zeros((3*N, 3))
    # W_mat = np.zeros((3*N, 3))
    for i in range(N):
        R_mat[3*i:3*i+3,:] = R[i]
        # W_mat[3*i:3*i+3,:] = W[i]
    # todo: generate W according to Gaussian Orthogonal Ensemble
    a = np.random.randn(3*N, 3*N)
    W = np.tril(a) + np.tril(a, -1).T
    # W = W_mat @ W_mat.T
    H = Lambda / N * R_mat @ R_mat.T + 1/np.sqrt(3 * N) * W

    # print('Generated %d rotation matrices' % N)
    return R_mat, H


def generate_training_data_so3(N: int, Lambda: float, R: int) -> [np.ndarray, np.ndarray]:
    Rot_total = []
    H_total = []
    for r in range(R):
        Rot_mat, H  = generate_data_so3(N, Lambda)

        # #todo: Return Rot_mat inside generate data so3
        # Rot_mat = np.zeros((3 * N, 3))
        # for i in range(N):
        #     Rot_mat[3 * i:3 * i + 3, :] = Rot[i]

        Rot_total.append(Rot_mat)
        H_total.append(H)

    Rot_total = np.asarray(Rot_total)
    H_total = np.asarray(H_total)

    return Rot_total, H_total


def generate_training_data_so3_with_j_ambiguity(N: int, Lambda: float, R: int) -> [np.ndarray, np.ndarray]:
    Rot_total = []
    H_total = []
    for r in range(R):
        Rot_mat, H  = generate_data_so3(N, Lambda)

        H, _ = apply_j(H, p=0.5)

        # #todo: Return Rot_mat inside generate data so3
        # Rot_mat = np.zeros((3 * N, 3))
        # for i in range(N):
        #     Rot_mat[3 * i:3 * i + 3, :] = Rot[i]

        Rot_total.append(Rot_mat)
        H_total.append(H)

    Rot_total = np.asarray(Rot_total)
    H_total = np.asarray(H_total)

    return Rot_total, H_total


if __name__ == '__main__':
    print('Main script')
    N = 100
    Lambda_range = np.arange(0.1,3,0.1)
    correlations_pim = []
    correlations_ppm = []
    for Lambda in Lambda_range:
        R,H = generate_data_so3(N,Lambda)
        R_hat_pim = pim_so3(H)
        R_hat_ppm = ppm_so3(H)
        res_pim = squared_correlation(R, R_hat_pim)
        res_ppm = squared_correlation(R, R_hat_ppm)
        correlations_pim.append(res_pim)
        correlations_ppm.append(res_ppm)

    plt.plot(Lambda_range, correlations_pim, label='pim')
    plt.plot(Lambda_range, correlations_ppm, label='ppm')
    plt.xlabel('SNR')
    plt.ylabel('Squared Correlation')
    plt.title('SO(3) Synchronization')
    plt.legend()
    plt.show()
    print('finished')

