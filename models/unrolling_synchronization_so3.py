import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np

from common.math_utils import project_batch

# tf.config.run_functions_eagerly(True)

class ProjectionBlock(keras.layers.Layer):
    def __init__(self, hidden_size=32, hidden_layers=1):
        super(ProjectionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dense_layers = [Dense(self.hidden_size) for _ in range(self.hidden_layers)]
        self.bns = [keras.layers.BatchNormalization() for _ in range(self.hidden_layers)]
        self.output_layer = Dense(9)
        self.input_layer = Dense(128)
        self.input_bn = keras.layers.BatchNormalization()
        self.output_bn = keras.layers.BatchNormalization()


    def call(self, x):
        x_reshaped = tf.reshape(x, [-1,9])
        y = x_reshaped
        y_prev = y

        # y = self.input_layer(y)
        # y = self.input_bn(y)
        # # y = keras.activations.relu(y)
        # y = keras.layers.LeakyReLU()(y)
        # # y = keras.activations.tanh(y)

        for i in range(self.hidden_layers):
            y = self.dense_layers[i](y)
            y = self.bns[i](y)
            # y = keras.activations.relu(y)
            # y = keras.layers.LeakyReLU()(y)
            y = keras.activations.tanh(y)
            # if i > 0:
            #     y = y + y_prev
            y_prev = y
        y = self.output_layer(y)
        y = self.output_bn(y)
        y = keras.activations.tanh(y)

        # y_squared = tf.pow(y,2)
        # y_sum = tf.reduce_sum(y_squared,axis=-1,keepdims=True)
        # y = y / tf.sqrt(y_sum / 3)

        y = tf.reshape(y, tf.shape(x))

        y = y + x

        return y

class StrictProjectionBlock(keras.layers.Layer):
    def __init__(self, num_layers=4):
        super(StrictProjectionBlock, self).__init__()
        self.num_layers = num_layers

    def call(self, x):
        x_reshaped = tf.reshape(x, [-1,3,3])
        norm = tf.sqrt(tf.reduce_sum(tf.pow(x_reshaped, 2), axis=[1, 2], keepdims=True))
        x_reshaped = x_reshaped / norm

        Q = x_reshaped
        for i in range(self.num_layers):
            N = tf.matmul(tf.transpose(Q, perm=[0, 2, 1]), Q)
            P = 1 / 2 * tf.matmul(Q, N)
            Q = 2 * Q + tf.matmul(P, N) - 3 * P

        Q_reshaped = tf.reshape(Q, tf.shape(x))
        return Q_reshaped


class SynchronizationBlock(keras.layers.Layer):
    def __init__(self, N):
        super(SynchronizationBlock, self).__init__()
        self.N = N
        # self.project_block = ProjectionBlock(hidden_size=32, hidden_layers=3)
        # self.project_block = ProjectionBlock(hidden_size=16, hidden_layers=3)
        self.project_block = ProjectionBlock(hidden_size=16, hidden_layers=1)
        # self.project_block = ProjectionBlock(hidden_size=16, hidden_layers=3)
        # self.project_block = ProjectionBlock(hidden_size=128, hidden_layers=3)
        # self.project_block = ProjectionBlock(hidden_size=32, hidden_layers=5)
        # self.project_block = ProjectionBlock(hidden_size=128, hidden_layers=5)

    def call(self, Y, x, x_prev):
        x1 = tf.matmul(Y, x)
        x1 = self.project_block(x1)
        #todo: use x_prev
        return x1


# def loss_so3(y_true,y_pred):
#     loss = 1 - tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.math.pow(tf.matmul(tf.transpose(y_true,perm=[0, 2, 1]),y_pred),2),axis=2),axis=1)/(y_true.shape[1]/3 * np.sqrt(3))**2)
#     return loss

def loss_so3(y_true,y_pred):
    loss = 1 - tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.math.pow(tf.matmul(tf.transpose(y_true,perm=[0, 2, 1]),y_pred),2),axis=2),axis=1)/(y_true.shape[1]/3 * np.sqrt(3))**2)

    # y_reshaped = tf.reshape(y_pred, [-1,9])
    # y_squared = tf.pow(y_reshaped,2)
    # y_sum = tf.reduce_sum(y_squared,axis=-1,keepdims=False)
    # reg = tf.pow(tf.reduce_mean(y_sum,axis=0)-3,2)

    # y_reshaped = tf.reshape(y_pred, [-1,3,3])
    # # det = tf.linalg.det(tf.matmul(y_reshaped,tf.transpose(y_reshaped,perm=[0,2,1])))
    # # reg = tf.pow(det-1,2)
    # yyt = tf.matmul(y_reshaped,tf.transpose(y_reshaped,perm=[0,2,1]))
    # target = tf.expand_dims(tf.eye(3,3,dtype=np.float32),axis=0)
    # diff = yyt - target
    # reg = tf.reduce_mean(tf.reduce_sum(tf.pow(diff,2),axis=[1,2]))

    # reg2 = tf.reduce_mean(tf.pow(tf.linalg.det(yyt) - 1,2))
    #
    # loss = loss + reg
    # loss = loss + reg + reg2

    return loss

def BuildModel(N,Lambda,DEPTH):
    v_in = keras.layers.Input((3*N, 3))
    v_in2 = keras.layers.Input((3*N, 3))
    Y = keras.layers.Input((3*N, 3*N))

    v = v_in
    v_prev = v_in2
    v_new = []

    # sb = SynchronizationBlock(N)
    for i in range(DEPTH):
        v_new = SynchronizationBlock(N)(Y, v, v_prev)
        # v_new = sb(Y, v, v_prev)
        v_prev = v
        v = v_new

    v_new = StrictProjectionBlock(num_layers=4)(v_new)

    model = Model(inputs=[v_in,v_in2, Y], outputs=v_new)
    # opt = keras.optimizers.Adam(learning_rate=0.001) # working 18:07
    # opt = keras.optimizers.Adam(learning_rate=0.005) # working 18:07
    # opt = keras.optimizers.Adam(learning_rate=0.1) # working 18:07
    opt = keras.optimizers.Adam(learning_rate=0.01) # working 18:07
    # opt = keras.optimizers.Adam(learning_rate=1) # working 18:07
    # opt = keras.optimizers.Adam(learning_rate=0.05) # working 18:07
    model.compile(optimizer=opt, loss=loss_so3)
    model.summary()
    return model


def EvaluateModel(model, Y, x, x_init, x_init2):
    x_est = model.predict([x_init, x_init2, Y])
    #apply projection on x_est

    x_est = project_batch(x_est)

    # loss = loss_so3(x.astype(np.float64), x_est.astype(np.float64))
    # loss = loss_so3(x, x_est)
    loss = loss_so3(x.astype(np.float32), x_est.astype(np.float32))
    print('[NN] loss = %lf' % loss)
    return x_est, loss

def TrainModel(model, Y, x, x_init, x_init2, Y_val,x_val,x_val_init,x_val_init2,epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)

    model.fit(x=[x_init,x_init2, Y],
              y=x.astype(np.float32),
              epochs=epochs,
              validation_data=([x_val_init,x_val_init2, Y_val], x_val.astype(np.float32)),
              validation_freq=20,
              callbacks=[tensorboard_callback],
              batch_size=128)
              # batch_size=5000)