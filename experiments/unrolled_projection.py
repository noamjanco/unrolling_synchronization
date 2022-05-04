import numpy as np
import scipy.linalg as linalg
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

def normalization_layer(M: np.ndarray) -> np.ndarray:
    M = M / np.sqrt(np.sum(np.power(M,2),axis=(0,1)))
    return M

def iterative_projection(M: np.ndarray, verbose: bool = False) -> np.ndarray:
    M = normalization_layer(M)

    Q = M
    Q_prev = np.ones_like(Q)
    for i in range(100):
        N = Q.T @ Q
        P = 1 / 2 * Q @ N
        Q = 2 * Q + P @ N - 3 * P
        err = min(linalg.norm(Q - Q_prev), linalg.norm(Q + Q_prev))
        if verbose:
            print('iter: ', i, ' err: ', err)
        if err < 1e-7:
            break
        Q_prev = Q
    return Q

sigma_range = np.arange(0,1,0.05)[1:]
fail_rates = []
num_trials = 1000
for sigma in sigma_range:
    num_fails = 0
    for i in range(num_trials):
        try:
            M = sigma*np.random.randn(3,3)
            # Q_sqrt = M @ linalg.inv(linalg.sqrtm(M.T @ M))
            # Q_sqrt = np.linalg.det(Q_sqrt) * Q_sqrt
            Q = iterative_projection(M)
            # err = min(linalg.norm(Q - Q_sqrt), linalg.norm(Q + Q_sqrt))
            # print(' err: ', err)
        except Exception as E:
            num_fails += 1
    fail_rate = num_fails / num_trials
    fail_rates.append(fail_rate)

plt.plot(sigma_range,fail_rates)
plt.xlabel('sigma')
plt.ylabel('fail rate')
plt.show()
# M = 0.2*np.random.randn(3,3)
# clip_val = 0.5
# M[M > clip_val] = clip_val
# M[M < -clip_val] = -clip_val
# M = np.eye(3) + 0.1*np.random.randn(3,3)
# M_squared = np.power(M,2)
# M_sum = np.sum(M_squared,axis=-1,keepdims=True)
# M = M / np.sqrt(M_sum / 3)

# print('condition number of M is:',np.linalg.cond(M))
# print('M min entry: ',np.min(M), 'M max entry: ', np.max(M))
# Q_svd = project_to_orthogonal_matrix(M, flip=True)
# Q_sqrt = M @ linalg.inv(linalg.sqrtm(M.T @ M))
# Q_sqrt = np.linalg.det(Q_sqrt) * Q_sqrt
#
# print('err between Q_sqrt and Q_svd: ', np.linalg.norm(Q_svd-Q_sqrt))
#
#
# # Q = M
# # errs = []
# # for i in range(100):
# #     N = Q.T @ Q
# #     P = 1/2 * Q @ N
# #     Q = 2*Q + P @ N - 3 * P
# #     err = min(linalg.norm(Q - Q_svd), linalg.norm(Q + Q_svd))
# #     errs.append(err)
# #     print('iter: ', i,' err: ',err)
# #     if err < 1e-7:
# #         break
# #     # print(Q)
#
# plt.plot(errs)
# plt.xlabel('iter')
# plt.ylabel('err')