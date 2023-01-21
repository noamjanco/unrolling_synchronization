import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np

from common.math_utils import project_batch

import os, pickle
import pandas as pd
import time

# tf.config.run_functions_eagerly(True)
from models.unrolling_synchronization_z_over_2 import loss_z_over_2


# class ProjectionBlock(keras.layers.Layer):
#     def __init__(self, hidden_size=32, hidden_layers=1):
#         super(ProjectionBlock, self).__init__()
#         self.hidden_size = hidden_size
#         self.hidden_layers = hidden_layers
#         self.dense_layers = [Dense(self.hidden_size) for _ in range(self.hidden_layers)]
#         self.bns = [keras.layers.BatchNormalization() for _ in range(self.hidden_layers)]
#         self.output_layer = Dense(9)
#         self.input_layer = Dense(128)
#         self.input_bn = keras.layers.BatchNormalization()
#         self.output_bn = keras.layers.BatchNormalization()
#
#
#     def call(self, x):
#         x_reshaped = tf.reshape(x, [-1,9])
#         y = x_reshaped
#         y_prev = y
#
#         for i in range(self.hidden_layers):
#             y = self.dense_layers[i](y)
#             y = self.bns[i](y)
#             y = keras.activations.relu(y)
#             y_prev = y
#         y = self.output_layer(y)
#         y = self.output_bn(y)
#         y = keras.activations.tanh(y)
#
#         y = tf.reshape(y, tf.shape(x))
#
#         return y

# class StrictProjectionBlock(keras.layers.Layer):
#     def __init__(self, num_layers=4):
#         super(StrictProjectionBlock, self).__init__()
#         self.num_layers = num_layers
#
#     def call(self, x):
#         x_reshaped = tf.reshape(x, [-1,3,3])
#         norm = tf.sqrt(tf.reduce_sum(tf.pow(x_reshaped, 2), axis=[1, 2], keepdims=True))
#         x_reshaped = x_reshaped / norm
#
#         Q = x_reshaped
#         for i in range(self.num_layers):
#             N = tf.matmul(tf.transpose(Q, perm=[0, 2, 1]), Q)
#             P = 1 / 2 * tf.matmul(Q, N)
#             Q = 2 * Q + tf.matmul(P, N) - 3 * P
#
#         Q_reshaped = tf.reshape(Q, tf.shape(x))
#         return Q_reshaped


# class SynchronizationBlock(keras.layers.Layer):
#     def __init__(self, N):
#         super(SynchronizationBlock, self).__init__()
#         self.N = N
#         self.project_block = ProjectionBlock(hidden_size=32, hidden_layers=1) # last best
#         self.project_block2 = ProjectionBlock(hidden_size=9, hidden_layers=1) # last best
#
#     def call(self, Y, x, x_prev):
#         x1 = tf.matmul(Y, x)
#         x1 = self.project_block(x1)
#         x1 = x1 + self.project_block2(x_prev)
#         #todo: use x_prev
#         return x1

class JConfigurationErrorBlock(keras.layers.Layer):
    def __init__(self, N, batchsize, global_indices):
        super(JConfigurationErrorBlock, self).__init__()
        self.learned_eye = tf.Variable(tf.eye(3),trainable=False)
        # self.learned_eye = tf.eye(3)
        self.J_block = tf.expand_dims(tf.linalg.diag(tf.tile((1., 1., -1.), [N])), axis=0)
        self.global_indices = global_indices
        self.batchsize = batchsize
        self.N = N

    def get_config(self):
        config = super().get_config()
        config.update({
            "N": self.N,
            "batchsize": self.batchsize,
            "global_indices": self.global_indices,
        })
        return config

    def call(self, H):
        H_j_conj = self.J_block @ H @ self.J_block

        H_ij = tf.reshape(tf.gather_nd(H, self.global_indices.ijs), (-1, 3, 3))
        H_ij_j_conj = tf.reshape(tf.gather_nd(H_j_conj, self.global_indices.ijs), (-1, 3, 3))
        H_jk = tf.reshape(tf.gather_nd(H, self.global_indices.jks), (-1, 3, 3))
        H_jk_j_conj = tf.reshape(tf.gather_nd(H_j_conj, self.global_indices.jks), (-1, 3, 3))
        H_ki = tf.reshape(tf.gather_nd(H, self.global_indices.kis), (-1, 3, 3))
        H_ki_j_conj = tf.reshape(tf.gather_nd(H_j_conj, self.global_indices.kis), (-1, 3, 3))

        j_comb = tf.stack([H_ij @ H_jk @ H_ki,
                           H_ij @ H_jk @ H_ki_j_conj,
                           H_ij @ H_jk_j_conj @ H_ki,
                           H_ij @ H_jk_j_conj @ H_ki_j_conj,
                           H_ij_j_conj @ H_jk @ H_ki,
                           H_ij_j_conj @ H_jk @ H_ki_j_conj,
                           H_ij_j_conj @ H_jk_j_conj @ H_ki,
                           H_ij_j_conj @ H_jk_j_conj @ H_ki_j_conj], axis=1)

        err = tf.linalg.norm(j_comb - self.learned_eye, axis=[2, 3])
        # err = tf.reduce_sum(tf.pow(j_comb - self.learned_eye,2), axis=[2, 3])

        return tf.reshape(err, (self.batchsize, -1, 8))


class SigmaBlock(keras.layers.Layer):
    def __init__(self, N, global_indices):
        super(SigmaBlock, self).__init__()
        self.shape = (int((N - 1) * N / 2), int((N - 1) * N / 2))
        self.global_indices = global_indices
        self.N = N
        stddev = 5
        self.hidden_1 = Dense(256, activation='relu',kernel_initializer=tf.keras.initializers.ones())
        # self.hidden_2 = Dense(8, activation='relu',kernel_initializer=tf.keras.initializers.ones())
        self.dense_1 = Dense(1, activation='tanh',kernel_initializer=tf.keras.initializers.ones())
        self.dense_2 = Dense(1, activation='tanh',kernel_initializer=tf.keras.initializers.ones())
        self.dense_3 = Dense(1, activation='tanh',kernel_initializer=tf.keras.initializers.ones())

    def get_config(self):
        config = super().get_config()
        config.update({
            "N": self.N,
            "global_indices": self.global_indices,
        })
        return config

    def call(self, err):
        # #todo: replace with MLP
        # min_ind = tf.argmin(err, axis=-1)
        # mu_ij_opt = tf.cast(tf.math.floormod(tf.floor(min_ind / 4), 2),tf.float32)
        # mu_jk_opt = tf.cast(tf.math.floormod(tf.floor(min_ind / 2), 2),tf.float32)
        # mu_ki_opt = tf.cast(tf.math.floormod(tf.floor(min_ind / 1), 2),tf.float32)
        # d1 = mu_ij_opt - mu_jk_opt
        # d2 = mu_jk_opt - mu_ki_opt
        # d3 = mu_ki_opt - mu_ij_opt

        hidden_2 = self.hidden_1(err)
        # hidden_2 = self.hidden_2(hidden_1)
        d1 = tf.squeeze(self.dense_1(hidden_2),axis=-1)
        d2 = tf.squeeze(self.dense_2(hidden_2),axis=-1)
        d3 = tf.squeeze(self.dense_3(hidden_2),axis=-1)

        res1 = tf.map_fn(lambda x: tf.scatter_nd(indices=self.global_indices._ij_jk, updates=x, shape=self.shape),
                         tf.pow(-1., d1))
        res2 = tf.map_fn(lambda x: tf.scatter_nd(indices=self.global_indices._jk_ki, updates=x, shape=self.shape),
                         tf.pow(-1., d2))
        res3 = tf.map_fn(lambda x: tf.scatter_nd(indices=self.global_indices._ki_ij, updates=x, shape=self.shape),
                         tf.pow(-1., d3))
        Sigma = res1 + res2 + res3
        Sigma = Sigma + tf.transpose(Sigma, perm=[0, 2, 1])
        return Sigma


class SynchronizationLayer(keras.layers.Layer):
    def __init__(self):
        super(SynchronizationLayer, self).__init__()

    def call(self, Y, x):
        x1 = tf.matmul(Y, x)
        # x_new = tf.divide(x1,(tf.sqrt(tf.reduce_sum(tf.pow(x1, 2), axis=-1, keepdims=True)))+1e-8)
        x_new = tf.divide(x1,tf.expand_dims(tf.linalg.norm(x1,axis=1),axis=-1))
        # x_new = tf.divide(x1,tf.expand_dims(tf.linalg.norm(x1,axis=1)+1e-1,axis=-1))

        return x_new


class CorrectJAmbiguityBlock(keras.layers.Layer):
    def __init__(self, global_indices):
        super(CorrectJAmbiguityBlock, self).__init__()
        self.global_indices = global_indices

    def get_config(self):
        config = super().get_config()
        config.update({
            "global_indices": self.global_indices,
        })
        return config

    def call(self, H, u_s):
        u_s_gather = tf.cast(tf.reshape(tf.less(tf.gather_nd(u_s, self.global_indices.u_s_gather_idx), 0), (H.shape[0], -1)), H.dtype)
        H_gather = tf.reshape(tf.gather_nd(H, self.global_indices.gather_idx), (H.shape[0], -1, 3, 3))
        H_gather2 = tf.reshape(tf.gather_nd(H, self.global_indices.gather_idx2), (H.shape[0], -1, 3, 3))
        without_j = tf.repeat(
            tf.expand_dims(tf.repeat(tf.expand_dims(tf.eye(3), axis=0), u_s_gather.shape[1], axis=0), axis=0),
            u_s_gather.shape[0], axis=0)
        with_j = tf.repeat(
            tf.expand_dims(
                tf.repeat(tf.expand_dims(tf.linalg.diag((1., 1., -1.)), axis=0), u_s_gather.shape[1], axis=0),
                axis=0), u_s_gather.shape[0], axis=0)
        indicator = tf.transpose(
            tf.repeat(tf.expand_dims(tf.transpose(tf.repeat(tf.expand_dims(u_s_gather, axis=-1), 3, axis=-1)), axis=0),
                      3,
                      axis=0))
        J_mat = indicator * with_j + (1 - indicator) * without_j

        H_conj = J_mat @ H_gather @ J_mat
        H2_conj = J_mat @ H_gather2 @ J_mat

        H1_scatter = tf.scatter_nd(self.global_indices.gather_idx, tf.reshape(H_conj, -1), shape=H.shape)
        H2_scatter = tf.scatter_nd(self.global_indices.gather_idx2, tf.reshape(H2_conj, -1), shape=H.shape)

        H = H1_scatter + H2_scatter
        return H


class IndexGeneration:
    def __init__(self, N, batchsize):
        """
        Generate indices used for gather / scatter operations
        :param N: Number of relative rotations
        :param batchsize: Number of samples in batch
        :return: None
        """
        super(IndexGeneration, self).__init__()
        self.N = N
        self.batchsize = batchsize
        pair_dict = {}
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                pair_dict[(i, j)] = idx
                pair_dict[(j, i)] = idx
                idx += 1

        self._ijs = []
        self._jks = []
        self._kis = []
        self._ij_jk = []
        self._jk_ki = []
        self._ki_ij = []
        indices = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    self._ijs.append(pair_dict[(i, j)])
                    self._jks.append(pair_dict[(j, k)])
                    self._kis.append(pair_dict[(k, i)])
                    self._ij_jk.append((self._ijs[-1], self._jks[-1]))
                    self._jk_ki.append((self._jks[-1], self._kis[-1]))
                    self._ki_ij.append((self._kis[-1], self._ijs[-1]))
                    indices.append((i, j, k))

        self.ijs = []
        self.jks = []
        self.kis = []
        for b in range(batchsize):
            for i, idx in enumerate(indices):
                for x in range(3):
                    for y in range(3):
                        self.ijs.append([b, 3 * idx[0] + x, 3 * idx[1] + y])
                        self.jks.append([b, 3 * idx[1] + x, 3 * idx[2] + y])
                        self.kis.append([b, 3 * idx[2] + x, 3 * idx[0] + y])

        self.ijs = tf.constant(np.asarray(self.ijs))
        self.jks = tf.constant(np.asarray(self.jks))
        self.kis = tf.constant(np.asarray(self.kis))

        self.gather_idx = []
        self.gather_idx2 = []
        self.u_s_gather_idx = []
        for b in range(batchsize):
            for i in range(N):
                for j in range(i + 1, N):
                    # todo: this is just increasing (range)
                    self.u_s_gather_idx.append([b, pair_dict[(i, j)], 0])
                    for x in range(3):
                        for y in range(3):
                            self.gather_idx.append([b, 3 * i + x, 3 * j + y])
                            self.gather_idx2.append([b, 3 * j + x, 3 * i + y])

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "N": self.N,
    #         "batchsize": self.batchsize,
    #     })
    #     return config
                        # def loss_so3(y_true,y_pred):
#     loss = 1 - tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.math.pow(tf.matmul(tf.transpose(y_true,perm=[0, 2, 1]),y_pred),2),axis=2),axis=1)/(y_true.shape[1]/3 * np.sqrt(3))**2)
#
#     return loss

def BuildModel(N, DEPTH, batchsize):
    v_in = keras.layers.Input((int((N - 1) * N / 2), 1))
    # v_in2 = keras.layers.Input((int((N - 1) * N / 2), 1))
    Y = keras.layers.Input((3*N, 3*N))

    global_indices = IndexGeneration(N, batchsize)
    # global_indices = generate_indices(N, batchsize)

    j_configuration_error = JConfigurationErrorBlock(N, batchsize, global_indices)(Y)

    sigma = SigmaBlock(N, global_indices)(j_configuration_error)

    v = v_in

    for i in range(DEPTH):
        v_new = SynchronizationLayer()(sigma, v)
        v = v_new

    v = tf.sign(v)

    # todo: in the next step use this
    # V_without_j_conj = CorrectJAmbiguityBlock(global_indices)(V, v)

    model = Model(inputs=[v_in, Y], outputs=v)

    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss=loss_z_over_2)
    model.summary()
    return model


def EvaluateModel(model, H, j_gt, j_init):
    j_est = model.predict([j_init, H])
    j_est = tf.math.sign(j_est)
    loss = loss_z_over_2(j_gt, j_est)
    return j_est, loss


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self._log_dir = log_dir
    def on_epoch_end(self, epoch, logs=None):
        filename = os.path.join(self._log_dir,'losses.pickle')
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                d = pickle.load(file)
        else:
            d = pd.DataFrame()
        logs['time'] = time.time()
        logs['epoch'] = epoch
        d = d.append(logs,ignore_index=True)
        with open(filename, 'wb') as file:
            pickle.dump(d, file)
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

def TrainModel(model, Y, j_gt, j_init, Y_val, j_gt_val, j_init_val, epochs, batchsize):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)

    checkpoint_filepath = 'tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.fit(x=[j_init, Y],
              y=j_gt.astype(np.float32),
              epochs=epochs,
              validation_data=([j_init_val, Y_val], j_gt_val.astype(np.float32)),
              validation_freq=20,
              # callbacks=[tensorboard_callback, model_checkpoint_callback, CustomCallback(log_dir)],
              # callbacks=[tensorboard_callback, model_checkpoint_callback],
              callbacks=[tensorboard_callback],
              batch_size=batchsize)
              # batch_size=5000)

    # The model weights (that are considered the best) are loaded into the model.
    # model.load_weights(checkpoint_filepath)
