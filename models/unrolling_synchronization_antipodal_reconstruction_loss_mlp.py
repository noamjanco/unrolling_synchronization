import keras
import tensorflow as tf
from keras.layers import Dense
from keras import Model
import datetime
import numpy as np

from models.unrolling_synchronization_antipodal_mlp import SingleLayer, CustomCallback



def loss_z_over_2(y_true,y_pred):
    loss = 1 - tf.reduce_mean(tf.math.abs(tf.math.reduce_sum(tf.math.multiply(y_true,y_pred),axis=1)))/y_true.shape[1]
    return loss

def reconstruction_loss_z_over_2(y_true,y_pred):
    loss = tf.reduce_mean(tf.math.minimum(tf.reduce_mean(tf.pow(y_true-y_pred,2),axis=1),tf.reduce_mean(tf.pow(y_true+y_pred,2),axis=1)))
    return loss

def BuildModel(L,N,Lambda,DEPTH):
    v_in = keras.layers.Input((N, 1))
    v_in2 = keras.layers.Input((N, 1))
    Y = keras.layers.Input((N, N))
    y = keras.layers.Input((L, N)) #complete data matrix

    y_reshaped = tf.reshape(y, (tf.shape(y)[0],L*N))
    hidden_size = L

    x = SingleLayer(output_size=hidden_size)(y_reshaped)
    for i in range(DEPTH):
        x = SingleLayer(output_size=hidden_size)(x)

    x_est = SingleLayer(output_size=L)(x)
    x_est = tf.expand_dims(x_est,axis=-1)


    model = Model(inputs=[v_in,v_in2, Y, y], outputs=x_est)
    opt = keras.optimizers.Adam(learning_rate=0.001) # working 18:07
    # model.compile(optimizer=opt, loss=loss_z_over_2)
    model.compile(optimizer=opt, loss=reconstruction_loss_z_over_2)
    model.summary()
    return model


def EvaluateModel(model, Y, y, x, x_init, x_init2):
    x_est = model.predict([x_init, x_init2, Y, y])
    # x_est = tf.math.sign(x_est)
    loss = reconstruction_loss_z_over_2(x, x_est)
    # loss = loss_z_over_2(x, x_est)
    print('[NN] loss = %lf' % loss)
    return x_est, loss

def TrainModel(model, Y,y, x, x_init, x_init2, Y_val,y_val, x_val, x_val_init,x_val_init2,epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                          write_images=True, profile_batch=0)
    model.fit(x=[x_init, x_init2, Y, y],
              y=x.astype(np.float32),
              epochs=epochs,
              validation_data=([x_val_init, x_val_init2, Y_val, y_val], x_val.astype(np.float32)),
              # validation_freq=20,
              callbacks=[tensorboard_callback, CustomCallback(log_dir)],
              batch_size=128)