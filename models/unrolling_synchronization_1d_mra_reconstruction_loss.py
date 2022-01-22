import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np


class NonLinearActivation(keras.layers.Layer):
    def __init__(self):
        super(NonLinearActivation, self).__init__()
        self.dense_1 = Dense(1,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.dense_3 = Dense(1,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
        self.dense_5 = Dense(1,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    def call(self, x):
        y = self.dense_1(x) + self.dense_3(tf.math.pow(x,3)) + self.dense_5(tf.math.pow(x,5))
        y = tf.math.minimum(y, tf.ones_like(y))
        y = tf.math.maximum(y, -tf.ones_like(y))
        return y


class SynchronizationBlock(keras.layers.Layer):
    def __init__(self, Lambda,N):
        super(SynchronizationBlock, self).__init__()
        self.dense_3 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        self.Lambda = Lambda
        self.nonlinear = NonLinearActivation()


    def call(self, Y_r, Y_i, x_r, x_i, x_prev_r, x_prev_i):

        x1_r = self.Lambda * (tf.matmul(Y_r, x_r) - tf.matmul(Y_i, x_i))
        x1_i = self.Lambda * (tf.matmul(Y_r, x_i) + tf.matmul(Y_i, x_r))
        x1_r = self.dense_3(x1_r)
        x1_i = self.dense_3(x1_i)
        x_abs_squared = tf.math.pow(x_r, 2) + tf.math.pow(x_i, 2)
        x2_r = - self.Lambda ** 2 * tf.math.multiply(tf.expand_dims(1 - tf.reduce_mean(x_abs_squared,axis=1),axis=-1), x_prev_r)
        x2_i = - self.Lambda ** 2 * tf.math.multiply(tf.expand_dims(1 - tf.reduce_mean(x_abs_squared,axis=1),axis=-1), x_prev_i)
        x1_r = x1_r + x2_r
        x1_i = x1_i + x2_i
        x1_abs = tf.math.sqrt(tf.math.pow(x1_r,2)+tf.math.pow(x1_i,2))
        x1_abs_denom = tf.math.maximum(x1_abs,1e-12*tf.ones_like(x1_abs))
        q = tf.math.divide(self.nonlinear(x1_abs), x1_abs_denom)
        x_new_r = tf.math.multiply(x1_r,q)
        x_new_i = tf.math.multiply(x1_i,q)


        return x_new_r, x_new_i


def loss_u_1(y_true,y_pred):
    y_pred_r, y_pred_i = tf.split(y_pred, 2, axis=-1)
    y_true_r, y_true_i = tf.split(y_true, 2, axis=-1)
    p_r = tf.math.reduce_sum(tf.math.multiply(y_true_r,y_pred_r) + tf.math.multiply(y_true_i,y_pred_i),axis=1)
    p_i = tf.math.reduce_sum(tf.math.multiply(y_true_r,y_pred_i) - tf.math.multiply(y_true_i,y_pred_r),axis=1)
    loss = 1 - tf.reduce_mean(tf.math.sqrt(tf.math.pow(p_r, 2) + tf.math.pow(p_i, 2)))/y_true.shape[1]
    return loss

def loss_u_1_complex(y_true,y_pred):
    y_true_r = y_true.real
    y_true_i = y_true.imag
    y_pred_r = y_pred.real
    y_pred_i = y_pred.imag
    p_r = tf.math.reduce_sum(tf.math.multiply(y_true_r, y_pred_r) + tf.math.multiply(y_true_i, y_pred_i), axis=1)
    p_i = tf.math.reduce_sum(tf.math.multiply(y_true_r, y_pred_i) - tf.math.multiply(y_true_i, y_pred_r), axis=1)
    loss = 1 - tf.reduce_mean(tf.math.sqrt(tf.math.pow(p_r, 2) + tf.math.pow(p_i, 2))) / y_true.shape[1]
    return loss

def reconstruction_loss_u_1(y_true,y_pred):
    y_a_r_mean, y_a_i_mean = tf.split(y_pred, 2, axis=-1)
    y_true_r, y_true_i = tf.split(y_true, 2, axis=-1)

    P = 10
    e_orbit_r = tf.math.cos(2 * np.pi * tf.expand_dims(tf.range(y_true_r.shape[1], dtype=tf.float32), axis=-1) @ tf.expand_dims(
        1/P*tf.range(P*y_true_r.shape[1], dtype=tf.float32), axis=0)/y_true_r.shape[1])

    e_orbit_i = tf.math.sin(
        2 * np.pi * tf.expand_dims(tf.range(y_true_r.shape[1], dtype=tf.float32), axis=-1) @ tf.expand_dims(
            1 / P *tf.range(P*y_true_r.shape[1], dtype=tf.float32), axis=0) / y_true_r.shape[1])

    y_orbit_r = tf.expand_dims(y_a_r_mean,axis=-1) * e_orbit_r - tf.expand_dims(y_a_i_mean,axis=-1) * e_orbit_i
    y_orbit_i = tf.expand_dims(y_a_i_mean,axis=-1) * e_orbit_r + tf.expand_dims(y_a_r_mean,axis=-1) * e_orbit_i


    loss = tf.reduce_mean(tf.reduce_min(tf.reduce_mean(tf.pow(y_orbit_r - tf.expand_dims(tf.cast(y_true_r,tf.float32),axis=-1),2)+
                                        tf.pow(y_orbit_i - tf.expand_dims(tf.cast(y_true_i,tf.float32),axis=-1),2),axis=1),axis=-1) / y_true.shape[1])

    return loss

def BuildModel(L, N, Lambda, DEPTH):
    v_in_r = keras.layers.Input((N, 1))
    v_in_i = keras.layers.Input((N, 1))
    v_in2_r = keras.layers.Input((N, 1))
    v_in2_i = keras.layers.Input((N, 1))
    Y_r = keras.layers.Input((N, N))
    Y_i = keras.layers.Input((N, N))
    y_r = keras.layers.Input((L, N))  # complete data matrix
    y_i = keras.layers.Input((L, N))  # complete data matrix
    x_r = v_in_r
    x_i = v_in_i
    x_prev_r = v_in2_r
    x_prev_i = v_in2_i

    for i in range(DEPTH):
        x_new_r, x_new_i = SynchronizationBlock(Lambda,N)(Y_r, Y_i, x_r, x_i, x_prev_r, x_prev_i)
        x_prev_r = x_r
        x_prev_i = x_i
        x_r = x_new_r
        x_i = x_new_i

    x_abs = tf.math.sqrt(tf.math.pow(x_r, 2) + tf.math.pow(x_i, 2))
    x_abs = tf.math.maximum(x_abs, 1e-12 * tf.ones_like(x_abs))

    x_r = tf.math.divide(x_r, x_abs)
    x_i = tf.math.divide(x_i, x_abs)
    s_est = tf.math.atan2(x_i, x_r) / (2 * np.pi) * L

    m = 2 * np.pi * s_est @ tf.expand_dims(tf.range(y_r.shape[1], dtype=tf.float32), axis=0) / y_r.shape[1]
    m = tf.transpose(m, perm=[0, 2, 1])
    e_r = tf.math.cos(m)
    e_i = tf.math.sin(m)
    y_a_r = e_r * y_r - e_i * y_i
    y_a_i = e_r * y_i + e_i * y_r
    y_a_r_mean = tf.reduce_mean(y_a_r, axis=-1)
    y_a_i_mean = tf.reduce_mean(y_a_i, axis=-1)
    x_est = tf.concat([y_a_r_mean, y_a_i_mean], axis=-1)

    model = Model(inputs=[v_in_r, v_in_i, v_in2_r, v_in2_i, Y_r, Y_i, y_r, y_i], outputs=x_est)
    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=opt, loss=reconstruction_loss_u_1)
    model.summary()
    return model


def EvaluateModel(model, Y, y, x, x_init, x_init2):
    x_est = model.predict([x_init.real,x_init.imag, x_init2.real, x_init2.imag, Y.real,Y.imag,y.real,y.imag])
    x_true = tf.concat([x.real.astype(np.float32),x.imag.astype(np.float32)],axis=-1)
    loss = reconstruction_loss_u_1(x_true,x_est)
    print('[NN] loss = %lf' % loss)
    return x_est, loss


def TrainModel(model, Y_r,Y_i,y,x, x_init_r,x_init_i, x_init2_r,x_init2_i,
               Y_val_r,Y_val_i,y_val,x_val, x_val_init_r,x_val_init_i,x_val_init2_r,x_val_init2_i,epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)

    # earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=True)

    # y = tf.concat([x_r, x_i], axis=-1)
    # y_val = tf.concat([x_val_r, x_val_i], axis=-1)
    model.fit(x=[x_init_r,x_init_i,x_init2_r,x_init2_i, Y_r,Y_i,y.real,y.imag],
              y=tf.concat([x.real,x.imag],axis=-1),
              epochs=epochs,
              # validation_data=([x_val_init_r,x_val_init_i,x_val_init2_r,x_val_init2_i, Y_val_r,Y_val_i,y_val], x_val),
              # callbacks=[tensorboard_callback, earlystopping_callback],
              validation_freq=20,
              callbacks=[tensorboard_callback],
              batch_size=100)