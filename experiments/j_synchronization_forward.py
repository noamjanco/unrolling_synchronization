from common.math_utils import rel_error_so3
from data_generation.generate_data_so3 import generate_data_so3, apply_j
from models.unrolling_synchronization_z_over_2 import loss_z_over_2
from synchronization.pim import pim_so3
import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.layers import Dense
from keras import Model

class PIMBlock(keras.layers.Layer):
    def __init__(self, Lambda, N):
        super(PIMBlock, self).__init__()
        self.Lambda = Lambda
        self.N = N

    def call(self, Y, x, x_prev, x_prev_prev):
        x1 = tf.matmul(Y, x)
        x_new = tf.divide(x1,(tf.sqrt(tf.reduce_sum(tf.pow(x1, 2), axis=-1, keepdims=True))))

        return x_new


def BuildModel(N, Lambda, DEPTH):
    v_in = keras.layers.Input((N, 1))
    v_in2 = keras.layers.Input((N, 1))
    Y = keras.layers.Input((N, N))

    v = v_in
    v_prev = v_in2
    v_prev_prev = v_in
    v_new = []

    for i in range(DEPTH):
        v_new = PIMBlock(Lambda, N)(Y, v, v_prev, v_prev_prev)
        v_prev_prev = v_prev
        v_prev = v
        v = v_new

    model = Model(inputs=[v_in, v_in2, Y], outputs=v_new)
    opt = keras.optimizers.Adam(learning_rate=0.001)  # working 18:07
    # opt = keras.optimizers.Adam(learning_rate=0.01) # working 18:07
    model.compile(optimizer=opt, loss=loss_z_over_2)
    model.summary()
    return model


def plot_with_confidence(x,y,std, label, fig, ax):
    x = np.asarray(x)
    y = np.asarray(y)
    base_line, = ax.plot(x, y, label=label)
    ax.fill_between(x, (y - std), (y + std), color=base_line.get_color(), alpha=.1)

def j_conj_err(Rot_est: np.ndarray, Rot: np.ndarray):
    J = np.diag((1,1,-1))
    err_reg = rel_error_so3(Rot_est, Rot)
    N = int(Rot_est.shape[0]/3)
    Rot_est_j_conj = np.zeros_like(Rot_est, dtype=np.float64)
    for i in range(N):
        Rot_est_j_conj[3*i:3*i+3,:] = J @ Rot_est[3*i:3*i+3,:] @ J
    err_j_conj = rel_error_so3(Rot_est_j_conj, Rot)
    return min(err_reg, err_j_conj)

def power_iteration(Sigma: np.ndarray, max_iterations=100, tol=1e-5) -> np.ndarray:
    v = 1e-3 * np.ones((Sigma.shape[0],1))
    i = 0
    while i < max_iterations:
        v_prev = v
        v = Sigma @ v_prev
        v = v / np.sqrt(np.sum(v ** 2))
        if np.sum((v - v_prev) ** 2) < tol:
            break
        i += 1
    return v

def calc_pair_dict(N):
    d = {}
    idx = 0
    for i in range(N):
        for j in range(i+1,N):
            d[(i,j)] = idx
            d[(j,i)] = idx
            idx +=1
    return d

def calc_err(H) -> tf.Tensor:
    """
    Calc error vector for each triplet
    :param H: relative rotation matrix of size (batch_size, 3*N, 3*N)
    :return: Error vector for each combination of J conjugation, for each triplet, of size (batch_size, num_triplets, 8)
    """
    N = int(H.shape[1]/3)
    batchsize = H.shape[0]
    J_block = tf.expand_dims(tf.linalg.diag(tf.tile((1., 1., -1.), [N])), axis=0)

    H_j_conj = J_block @ H @ J_block

    H_ij = tf.reshape(tf.gather_nd(H, ijs), (-1, 3, 3))
    H_ij_j_conj = tf.reshape(tf.gather_nd(H_j_conj, ijs), (-1, 3, 3))
    H_jk = tf.reshape(tf.gather_nd(H, jks), (-1, 3, 3))
    H_jk_j_conj = tf.reshape(tf.gather_nd(H_j_conj, jks), (-1, 3, 3))
    H_ki = tf.reshape(tf.gather_nd(H, kis), (-1, 3, 3))
    H_ki_j_conj = tf.reshape(tf.gather_nd(H_j_conj, kis), (-1, 3, 3))

    j_comb = tf.stack([H_ij @ H_jk @ H_ki,
                       H_ij @ H_jk @ H_ki_j_conj,
                       H_ij @ H_jk_j_conj @ H_ki,
                       H_ij @ H_jk_j_conj @ H_ki_j_conj,
                       H_ij_j_conj @ H_jk @ H_ki,
                       H_ij_j_conj @ H_jk @ H_ki_j_conj,
                       H_ij_j_conj @ H_jk_j_conj @ H_ki,
                       H_ij_j_conj @ H_jk_j_conj @ H_ki_j_conj], axis=1)

    err = tf.linalg.norm(j_comb - tf.eye(3), axis=[2,3])

    return tf.reshape(err, (batchsize, -1, 8))

def calc_mu(err_rs: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """ Calculate mu for each pair based on the error"""
    min_ind = tf.argmin(err_rs, axis=-1)
    mu_ij_opt = tf.math.floormod(tf.floor(min_ind / 4), 2)
    mu_jk_opt = tf.math.floormod(tf.floor(min_ind / 2), 2)
    mu_ki_opt = tf.math.floormod(min_ind / 1, 2)
    return mu_ij_opt, mu_jk_opt, mu_ki_opt

def build_sigma(mu_ij_opt: tf.Tensor, mu_jk_opt: tf.Tensor, mu_ki_opt: tf.Tensor) -> tf.Tensor:
    """
    Build the matrix sigma from mu_ij_opt, mu_jk_opt, mu_ki_opt
    :param mu_ij_opt:
    :param mu_jk_opt:
    :param mu_ki_opt:
    :return: Matrix sigma of size (batch_size , (N - 1) * N / 2 , (N - 1) * N / 2)
    """
    shape = (int((N - 1) * N / 2), int((N - 1) * N / 2))

    res1 = tf.map_fn(lambda x: tf.scatter_nd(indices=_ij_jk, updates=x, shape=shape),
                     tf.pow(-1., mu_ij_opt - mu_jk_opt))
    res2 = tf.map_fn(lambda x: tf.scatter_nd(indices=_jk_ki, updates=x, shape=shape),
                     tf.pow(-1., mu_jk_opt - mu_ki_opt))
    res3 = tf.map_fn(lambda x: tf.scatter_nd(indices=_ki_ij, updates=x, shape=shape),
                     tf.pow(-1., mu_ki_opt - mu_ij_opt))
    Sigma = res1 + res2 + res3
    Sigma = Sigma + tf.transpose(Sigma, perm=[0, 2, 1])
    return Sigma

def unroll_pim(sigma: tf.Tensor, depth: int = 100) -> tf.Tensor:
    """
    Perform unrolled power iteration on input sigma
    :param sigma: Input matrix of size (batch_size , (N - 1) * N / 2 , (N - 1) * N / 2)
    :param depth: how many iterations to unroll
    :return: Eigenvector of Sigma
    """
    # u_s = model.predict([np.asarray(x_init,dtype=np.float64), np.asarray(x_init,dtype=np.float64), np.asarray(Sigma,dtype=np.float64)])
    x_init = 1e-3*np.expand_dims(np.ones((sigma.shape[0],sigma.shape[1])),axis=-1)
    v = x_init
    for i in range(depth):
        v_new = sigma @ v
        v_new = tf.divide(v_new,tf.expand_dims(tf.linalg.norm(v_new,axis=1),axis=-1))
        v = v_new
    return v_new

def correct_j_ambiguity(H: tf.Tensor, u_s: tf.Tensor) -> tf.Tensor:
    """
    Correct J ambiguity of H according to the estimate u_s
    :param H: Input relative rotation matrix, of size (batch_size, 3*N, 3*N)
    :param u_s: Estimated eigenvector of Sigma, indicates which pair contains J ambiguity
    :return: H without J ambiguity
    """
    u_s_gather = tf.cast(tf.reshape(tf.less(tf.gather_nd(u_s, u_s_gather_idx), 0), (batchsize, -1)), tf.float32)
    H_gather = tf.reshape(tf.gather_nd(H, gather_idx), (batchsize, -1, 3, 3))
    H_gather2 = tf.reshape(tf.gather_nd(H, gather_idx2), (batchsize, -1, 3, 3))
    without_j = tf.repeat(
        tf.expand_dims(tf.repeat(tf.expand_dims(tf.eye(3), axis=0), u_s_gather.shape[1], axis=0), axis=0),
        u_s_gather.shape[0], axis=0)
    with_j = tf.repeat(
        tf.expand_dims(tf.repeat(tf.expand_dims(tf.linalg.diag((1., 1., -1.)), axis=0), u_s_gather.shape[1], axis=0),
                       axis=0), u_s_gather.shape[0], axis=0)
    indicator = tf.transpose(
        tf.repeat(tf.expand_dims(tf.transpose(tf.repeat(tf.expand_dims(u_s_gather, axis=-1), 3, axis=-1)), axis=0), 3,
                  axis=0))
    J_mat = indicator * with_j + (1 - indicator) * without_j

    H_conj = J_mat @ H_gather @ J_mat
    H2_conj = J_mat @ H_gather2 @ J_mat

    H1_scatter = tf.scatter_nd(gather_idx, tf.reshape(H_conj, -1), shape=H.shape)
    H2_scatter = tf.scatter_nd(gather_idx2, tf.reshape(H2_conj, -1), shape=H.shape)

    H = H1_scatter + H2_scatter
    return H

def j_synch_forward(H: np.ndarray):
    # calc error among different possibilities
    err_rs = calc_err(H)

    # calc mu
    mu_ij_opt, mu_jk_opt, mu_ki_opt = calc_mu(err_rs)

    # Build the matrix Sigma
    Sigma = build_sigma(mu_ij_opt, mu_jk_opt, mu_ki_opt)

    # Unroll PIM
    u_s = unroll_pim(Sigma, depth=100)

    # Correct J ambiguity
    H = correct_j_ambiguity(H, u_s)

    return H


np.random.seed(1)
# Lambda = 3
# N = 50
# Lambda_range = np.arange(1,10,1)
N = 20
R = 5
# Lambda_range = np.arange(1,9,2)
Lambda_range = np.arange(1,9,2)
# Lambda_range = [3.]
# Lambda_range = [9.]
# Lambda_range = [10]
err_no_j_acc = []
err_with_j_acc = []
err_with_j_synch_acc = []
err_no_j_vec = []
err_no_j_vec_std = []
err_with_j_vec =[]
err_with_j_vec_std =[]
err_with_j_synch_vec = []
err_with_j_synch_vec_std = []


# ---------------------------------------------------------------------------- #
# Fixed computation for N and batchsize
batchsize = R
pair_dict = calc_pair_dict(N)
_ijs = []
_jks = []
_kis = []
indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _ijs.append(pair_dict[(i, j)])
            _jks.append(pair_dict[(j, k)])
            _kis.append(pair_dict[(k, i)])
            indices.append((i, j, k))

_ij_jk = []
_jk_ki = []
_ki_ij = []
for i in range(len(_ijs)):
    _ij_jk.append((_ijs[i], _jks[i]))
    _jk_ki.append((_jks[i], _kis[i]))
    _ki_ij.append((_kis[i], _ijs[i]))

ijs = []
jks = []
kis = []
for b in range(batchsize):
    for i, idx in enumerate(indices):
        for x in range(3):
            for y in range(3):
                ijs.append([b, 3 * idx[0] + x, 3 * idx[1] + y])
                jks.append([b, 3 * idx[1] + x, 3 * idx[2] + y])
                kis.append([b, 3 * idx[2] + x, 3 * idx[0] + y])


ijs = tf.constant(np.asarray(ijs))
jks = tf.constant(np.asarray(jks))
kis = tf.constant(np.asarray(kis))

__ijs = []
for i in range(N):
    for j in range(i + 1, N):
        __ijs.append(pair_dict[(i, j)])

# model = BuildModel(int((N - 1) * N / 2),Lambda=1.,DEPTH=100)

gather_idx  = []
gather_idx2  = []
u_s_gather_idx = []
for b in range(batchsize):
    for i in range(N):
        for j in range(i + 1, N):
            #todo: this is just increasing (range)
            u_s_gather_idx.append([b,pair_dict[(i, j)],0])
            for x in range(3):
                for y in range(3):
                    gather_idx.append([b, 3 * i + x, 3 * j + y])
                    gather_idx2.append([b, 3 * j + x, 3 * i + y])
# ---------------------------------------------------------------------------- #


for Lambda in tqdm.tqdm(Lambda_range):
    err_no_j_acc = []
    err_with_j_acc = []
    err_with_j_synch_acc = []
    H_list = []
    Rot_list = []
    for r in range(R):
        # generate samples according to R_ij = {R_i^T R_j, J R_i^TR_jJ}, where J=diag(1,1,-1)
        Rot, H = generate_data_so3(N, Lambda)
        # H = H * N / Lambda
        # err with no J conj
        Rot_est = pim_so3(H)
        Rot_est = np.real(Rot_est)
        err_no_j = j_conj_err(Rot_est, Rot)
        err_no_j_acc.append(err_no_j)
        H, J_conj_dict = apply_j(H, p=0.5)

        # err with J conj without J-synch
        Rot_est = pim_so3(H)
        Rot_est = np.real(Rot_est)
        err_with_j = j_conj_err(Rot_est, Rot)
        err_with_j_acc.append(err_with_j)

        # error with J conj and J-synch
        # H = j_synchronization(H, J_conj_dict, N, Lambda)
        H_list.append(np.asarray(H,np.float32))
        Rot_list.append(Rot)

    H_new = j_synch_forward(np.asarray(H_list))
    for r in range(R):
        Rot_est = pim_so3(H_new[r])
        Rot_est = np.real(Rot_est)
        err_with_j_synch = j_conj_err(Rot_est, Rot_list[r])
        err_with_j_synch_acc.append(err_with_j_synch)

    err_no_j_vec.append(np.mean(err_no_j_acc))
    err_no_j_vec_std.append(np.std(err_no_j_acc))
    err_with_j_vec.append(np.mean(err_with_j_acc))
    err_with_j_vec_std.append(np.std(err_with_j_acc))
    err_with_j_synch_vec.append(np.mean(err_with_j_synch_acc))
    err_with_j_synch_vec_std.append(np.std(err_with_j_synch_acc))

fig, ax = plt.subplots()
plot_with_confidence(Lambda_range, err_no_j_vec,err_no_j_vec_std, 'err without J ambiguity', fig, ax)
plot_with_confidence(Lambda_range, err_with_j_vec, err_with_j_vec_std, 'err with J ambiguity', fig, ax)
plot_with_confidence(Lambda_range, err_with_j_synch_vec, err_with_j_synch_vec_std, 'err with J ambiguity using J-Synch', fig, ax)
plt.xlabel('$SNR$')
plt.ylabel('Alignment Error')
plt.title(f'Error comparison with N={N}')
plt.legend()
plt.show()
# print('err = ', err_pim)
