import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np


class NonLinearActivation(keras.layers.Layer):
    def __init__(self):
        super(NonLinearActivation, self).__init__()
        self.dense_1 = Dense(1,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
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

def BuildModel(N,Lambda,DEPTH):
    v_in_r = keras.layers.Input((N, 1))
    v_in_i = keras.layers.Input((N, 1))
    v_in2_r = keras.layers.Input((N, 1))
    v_in2_i = keras.layers.Input((N, 1))
    Y_r = keras.layers.Input((N, N))
    Y_i = keras.layers.Input((N, N))
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

    # v_new = tf.math.sign(v_new)
    x_abs = tf.math.sqrt(tf.math.pow(x_r, 2) + tf.math.pow(x_i, 2))
    x_abs = tf.math.maximum(x_abs, 1e-12 * tf.ones_like(x_abs))

    x_r = tf.math.divide(x_r, x_abs)
    x_i = tf.math.divide(x_i, x_abs)
    output = tf.concat([x_r, x_i], axis=-1)
    model = Model(inputs=[v_in_r,v_in_i,v_in2_r,v_in2_i, Y_r,Y_i], outputs=output)
    opt = keras.optimizers.Adam(learning_rate=0.005)

    model.compile(optimizer=opt, loss=loss_u_1)
    model.summary()
    return model


def EvaluateModel(model, Y, x, x_init, x_init2):
    x_est = model.predict([x_init.real,x_init.imag, x_init2.real, x_init2.imag, Y.real,Y.imag])
    x_est_norm = tf.math.sqrt(tf.reduce_sum(tf.math.pow(x_est, 2), axis=-1))
    x_est = x_est / np.expand_dims(x_est_norm, axis=-1)
    x_est_r, x_est_i = tf.split(x_est, 2, axis=-1)
    x_est_complex = x_est_r.numpy() + 1j * x_est_i.numpy()
    loss = loss_u_1_complex(x.astype(np.csingle), x_est_complex.astype(np.csingle))
    print('[NN] loss = %lf' % loss)
    return x_est, loss

def TrainModel(model, Y_r,Y_i, x_r,x_i, x_init_r,x_init_i, x_init2_r,x_init2_i,
               Y_val_r,Y_val_i,x_val_r,x_val_i,x_val_init_r,x_val_init_i,x_val_init2_r,x_val_init2_i,epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)
    y = tf.concat([x_r, x_i], axis=-1)
    y_val = tf.concat([x_val_r, x_val_i], axis=-1)
    model.fit(x=[x_init_r,x_init_i,x_init2_r,x_init2_i, Y_r,Y_i],
              y=y,
              epochs=epochs,
              validation_data=([x_val_init_r,x_val_init_i,x_val_init2_r,x_val_init2_i, Y_val_r,Y_val_i], y_val),
              validation_freq=20,
              callbacks=[tensorboard_callback],
              batch_size=100)