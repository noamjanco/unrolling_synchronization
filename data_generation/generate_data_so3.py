import numpy as np
import typing
from typing import List
import matplotlib.pyplot as plt


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


def generate_data_so3(N: int, Lambda: float) -> [List[np.ndarray], np.ndarray]:

    H = np.zeros((3*N, 3*N))
    R = [project_to_orthogonal_matrix(np.random.randn(3,3), flip=True) for _ in range(N)]

    # for i in range(N):
    #     for j in range(N):
    #         H[3*i:3*i+3,3*j:3*j+3] = R[i] @ R[j].T

    R_mat = np.zeros((3*N, 3))
    for i in range(N):
        R_mat[3*i:3*i+3,:] = R[i]
    # todo: generate W according to Gaussian Orthogonal Ensemble
    a = np.random.randn(3*N, 3*N)
    W = np.tril(a) + np.tril(a, -1).T
    H = Lambda / N * R_mat @ R_mat.T + 1/np.sqrt(3 * N) * W

    # print('Generated %d rotation matrices' % N)
    return R, H


def pim_so3(H: np.ndarray) -> List[np.ndarray]:
    # Extract N from the size of H
    N = int(H.shape[0] / 3)

    # Compute the eigenvectors of H
    w, v = np.linalg.eig(H)

    sort_idx = np.argsort(np.abs(w))

    # Take the first 3 eigenvectors
    R_rec = v[:,sort_idx[-3:]]

    # Project onto the orthogonal matrices
    R_rec_proj = [project_to_orthogonal_matrix(R_rec[3*i:3*i+3,:]) for i in range(N)]

    return R_rec_proj

def project(M: np.ndarray) -> np.ndarray:
    """
    This function projects each 3x3 block of M into the space of orthogonal matrices
    :param M: 3N x 3 Matrix
    :return: a projected matrix, 3N X 3
    """
    N = int(M.shape[0] / 3)
    M_proj = np.zeros_like(M)
    M_proj_list = [project_to_orthogonal_matrix(M[3*i:3*i+3,:]) for i in range(N)]
    for i in range(N):
        M_proj[3*i:3*i+3,:] = M_proj_list[i]
    return M_proj


def ppm_so3(H: np.ndarray, num_iterations: int = 200) -> List[np.ndarray]:
    # Extract N from the size of H
    N = int(H.shape[0] / 3)

    # Initialize the matrix z
    z = initialize_matrix(N)

    for i in range(num_iterations):
        z_tag = project(H @ z)
        if np.linalg.norm(z_tag-z) < 1e-3:
            break
        z = z_tag


    #todo: avoid this step by changing the return of PIM method
    z_tag_list = []
    for i in range(N):
        z_tag_list.append(z_tag[3*i:3*i+3,:])
    return z_tag_list

def squared_correlation(X, X_hat) -> float:

    N = len(X)
    X_mat = np.zeros((N*3,3))
    for i in range(N):
        X_mat[3*i:3*i+3,:] = X[i]
    X_hat_mat = np.zeros((N*3,3))
    for i in range(N):
        X_hat_mat[3*i:3*i+3,:] = X_hat[i]

    return np.linalg.norm(X_mat.T @ X_hat_mat) / (N * np.sqrt(3))

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

