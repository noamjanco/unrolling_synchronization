import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np


class NonLinearActivation(keras.layers.Layer):
    def __init__(self):
        super(NonLinearActivation, self).__init__()
        self.dense_1 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        self.dense_3 = Dense(1,kernel_initializer=tf.keras.initializers.zeros())
        self.dense_5 = Dense(1,kernel_initializer=tf.keras.initializers.zeros())

    def call(self, x):
        y = self.dense_1(x) + self.dense_3(tf.math.pow(x,3)) + self.dense_5(tf.math.pow(x,5))
        y = tf.math.minimum(y, tf.ones_like(y))
        y = tf.math.maximum(y, -tf.ones_like(y))
        return y

class NewNonLinearActivation(keras.layers.Layer):
    def __init__(self, hidden_size=32, hidden_layers=1):
        super(NewNonLinearActivation, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dense_layers = [Dense(self.hidden_size) for _ in range(self.hidden_layers)]
        self.bns = [keras.layers.BatchNormalization() for _ in range(self.hidden_layers)]
        self.output_layer = Dense(1)
        self.output_bn = keras.layers.BatchNormalization()


    def call(self, x):
        y = x

        for i in range(self.hidden_layers):
            y = self.dense_layers[i](y)
            y = self.bns[i](y)
            y = keras.activations.relu(y)

        y = self.output_layer(y)
        y = self.output_bn(y)
        y = keras.activations.tanh(y)

        return y

class NonLinearActivation2(keras.layers.Layer):
    def __init__(self):
        super(NonLinearActivation2, self).__init__()
        self.dense_1 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        self.dense_2 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        self.dense_3 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        self.dense_4 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        self.dense_5 = Dense(1,kernel_initializer=tf.keras.initializers.ones())

    def call(self, x):
        y = self.dense_1(x) + self.dense_2(tf.math.pow(x,2)) + self.dense_3(tf.math.pow(x,3)) + self.dense_4(tf.math.pow(x,4)) + self.dense_5(tf.math.pow(x,5))
        y = tf.math.minimum(y, tf.ones_like(y))
        y = tf.math.maximum(y, -tf.ones_like(y))
        return y

class SynchronizationBlock(keras.layers.Layer):
    def __init__(self, Lambda, N):
        super(SynchronizationBlock, self).__init__()
        self.dense_1 = Dense(1,kernel_initializer=tf.keras.initializers.ones())
        # self.dense_1 = Dense(1)
        self.Lambda = Lambda
        self.N = N
        # self.nonlinear = NonLinearActivation()
        self.nonlinear = NewNonLinearActivation(hidden_size=32,hidden_layers=1)
        # self.nonlinear2 = NonLinearActivation2()
        self.nonlinear2 = NewNonLinearActivation(hidden_size=32,hidden_layers=1)
        self.bn = keras.layers.BatchNormalization()

    def call(self, Y, x, x_prev, x_prev_prev):
        x1 = self.Lambda * tf.matmul(Y, x)
        x1 = self.dense_1(x1)
        x2 = - self.Lambda ** 2 * tf.math.multiply(tf.expand_dims(1 - tf.reduce_mean(tf.pow(self.nonlinear2(x),2),axis=1),axis=-1), x_prev)
        x_new = x1 + x2
        x_new = self.bn(x_new)
        x_new = self.nonlinear(x_new)

        return x_new


def loss_z_over_2(y_true,y_pred):
    loss = 1 - tf.reduce_mean(tf.math.abs(tf.math.reduce_sum(tf.math.multiply(y_true,y_pred),axis=1)))/y_true.shape[1]
    return loss


def BuildModel(N,Lambda,DEPTH):
    v_in = keras.layers.Input((N, 1))
    v_in2 = keras.layers.Input((N, 1))
    Y = keras.layers.Input((N, N))

    v = v_in
    v_prev = v_in2
    v_prev_prev = v_in
    v_new = []

    for i in range(DEPTH):
        v_new = SynchronizationBlock(Lambda,N)(Y, v, v_prev, v_prev_prev)
        v_prev_prev = v_prev
        v_prev = v
        v = v_new


    model = Model(inputs=[v_in,v_in2, Y], outputs=v_new)
    opt = keras.optimizers.Adam(learning_rate=0.001) # working 18:07
    # opt = keras.optimizers.Adam(learning_rate=0.01) # working 18:07
    model.compile(optimizer=opt, loss=loss_z_over_2)
    model.summary()
    return model


def EvaluateModel(model, Y, x, x_init, x_init2):
    x_est = model.predict([x_init, x_init2, Y])
    x_est = tf.math.sign(x_est)
    loss = loss_z_over_2(x, x_est)
    print('[NN] loss = %lf' % loss)
    return x_est, loss

def TrainModel(model, Y, x, x_init, x_init2, Y_val,x_val,x_val_init,x_val_init2,epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)

    checkpoint_filepath = 'tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    model.fit(x=[x_init,x_init2, Y],
              y=x.astype(np.float32),
              epochs=epochs,
              validation_data=([x_val_init,x_val_init2, Y_val], x_val.astype(np.float32)),
              validation_freq=20,
              callbacks=[tensorboard_callback, model_checkpoint_callback],
              batch_size=128)

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_filepath)