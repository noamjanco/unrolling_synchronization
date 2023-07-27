import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np
import os
import pandas as pd
import pickle
import time



class SingleLayer(keras.layers.Layer):
    def __init__(self, output_size=32):
        super(SingleLayer, self).__init__()
        self.dense_layer = Dense(output_size)
        self.bn = keras.layers.BatchNormalization()


    def call(self, x):
        y = self.dense_layer(x)
        y = self.bn(y)
        y = keras.activations.relu(y)
        # y = keras.activations.tanh(y)

        return y


def loss_z_over_2(y_true,y_pred):
    loss = 1 - tf.reduce_mean(tf.math.abs(tf.math.reduce_sum(tf.math.multiply(y_true,y_pred),axis=1)))/y_true.shape[1]
    return loss


def BuildModelMLP(N,Lambda,DEPTH):
    v_in = keras.layers.Input((N, 1))
    v_in2 = keras.layers.Input((N, 1))
    Y = keras.layers.Input((N, N))

    hidden_size = 20
    Y_reshaped = tf.reshape(Y, (tf.shape(Y)[0],N*N))
    # Y_reshaped = tf.squeeze(Y)
    x = SingleLayer(output_size=hidden_size)(Y_reshaped)
    for i in range(DEPTH):
        x = SingleLayer(output_size=hidden_size)(x)

    x = SingleLayer(output_size=N)(x)
    x = keras.activations.tanh(x)
    x = tf.expand_dims(x,axis=-1)
    model = Model(inputs=[v_in,v_in2, Y], outputs=x)
    opt = keras.optimizers.Adam(learning_rate=0.0001) # working 18:07
    model.compile(optimizer=opt, loss=loss_z_over_2)
    model.summary()
    return model


def EvaluateModelMLP(model, Y, x, x_init, x_init2):
    x_est = model.predict([x_init, x_init2, Y])
    x_est = tf.math.sign(x_est)
    loss = loss_z_over_2(x, x_est)
    print('[NN] loss = %lf' % loss)
    return x_est, loss

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

def TrainModelMLP(model, Y, x, x_init, x_init2, Y_val,x_val,x_val_init,x_val_init2,epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)
    model.fit(x=[x_init,x_init2, Y],
              y=x.astype(np.float32),
              epochs=epochs,
              validation_data=([x_val_init,x_val_init2, Y_val], x_val.astype(np.float32)),
              # validation_freq=2,
              callbacks=[tensorboard_callback, CustomCallback(log_dir)],
              # batch_size=128)
              batch_size=128)