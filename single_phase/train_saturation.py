
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import numpy as np
import h5py
import keras
import pickle
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adagrad, Adam, Nadam
from xlrd import open_workbook
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import mse
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Input, Concatenate, Lambda, Activation
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.layers.merge import add
from keras.constraints import Constraint
from keras.initializers import Initializer
import scipy.io
import tensorflow as tf
from tensorflow.python.ops import sparse_ops as sparse
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from uleft_velocity_layer import ULeftLayer
from uright_velocity_layer import URightLayer
from utop_velocity_layer import UTopLayer
from ubottom_velocity_layer import UBottomLayer

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b, permutation

def trianing_saturation( sparse = True):

    filename_xtr = './data/sat_train_x_600.csv'
    raw_data_xtr = open(filename_xtr, 'rt')
    x_raw_data = np.loadtxt(raw_data_xtr, delimiter=",")#, usecols=range(10081))

    filename_ytr = './data/sat_train_y_600.csv'
    raw_data_ytr = open(filename_ytr, 'rt')
    y_raw_data = np.loadtxt(raw_data_ytr, delimiter=",")#, usecols=range(81))

    x_train = x_raw_data*100
    y_train = y_raw_data*100
    print(x_train.shape, 'train samples')
    print(y_train.shape, 'train samples')

    filename_xte = './data/sat_test_x_600.csv'
    raw_data_xte = open(filename_xte, 'rt')
    x_test_data = np.loadtxt(raw_data_xte, delimiter=",")
    x_test = x_test_data*100

    filename_yte = './data/sat_test_y_600.csv'
    raw_data_yte = open(filename_yte, 'rt')
    y_test_data = np.loadtxt(raw_data_yte, delimiter=",")
    y_test = y_test_data*100


    epochs = 1000
    batch_size = 100
    alpha=0.3

    input_dim = x_train.shape[1]
    out_dim = input_dim
    input = Input(shape=(input_dim,), )
    if sparse:
        # separate 4 directions of u
        out1 = ULeftLayer(input_dim=input_dim, output_dim=out_dim)(input)
        out1 = LeakyReLU(0.3)(out1)#Activation('relu')(out1) #
        out2 = URightLayer(input_dim=input_dim, output_dim=out_dim)(input)
        out2 = LeakyReLU(0.3)(out2)#Activation('relu')(out2)  #
        out3 = UTopLayer(input_dim=input_dim, output_dim=out_dim)(input)
        out3 = LeakyReLU(0.3)(out3)#Activation('relu')(out3)  #
        out4 = UBottomLayer(input_dim=input_dim, output_dim=out_dim)(input)
        out4 = LeakyReLU(0.3)(out4)#Activation('relu')(out4)  #

        sum1 = add([out1, out2])
        sum2 = add([out3, sum1])
        sum3 = add([out4, sum2])

        out5 = add([sum3, input])
        output = Activation('linear')(out5)
    else:
        out5 = Dense(nuron)(input)
        out5 = LeakyReLU(0.2)(out5)
        out5 = Dense(out_dim)(input)
        output = LeakyReLU(0.2)(out5)

    model = Model(input, output)
    model.summary()

    admax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    weight_path = 'best_weights_1phsat_l53.hdf5'
    checkpointer = ModelCheckpoint(filepath=weight_path, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    hist_loss = LossHistory()

    model.compile(loss= mse,
                  optimizer= admax,
                  metrics=[mse])

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size = batch_size,
                        verbose=1,
                        callbacks = [checkpointer])#validation_data=(x_test, y_test)

    model_path = 'model_1phsat_l53.h5'
    model.save(model_path)
    # np.savetxt('los_hist_sat_1ph_800.csv', np.array(hist_loss.losses), delimiter=",")
    # np.savetxt('los_val_hist_sat_1ph_800.csv', np.array(history.history['val_loss']), delimiter=",")

def prediction_saturation(total_steps):
    weight_path = 'best_weights_1phsat_l53.hdf5'
    model_path = 'model_1phsat_l53.h5'
    model = load_model(model_path, custom_objects={'ULeftLayer': ULeftLayer, 'URightLayer': URightLayer,
                                                    'UTopLayer': UTopLayer, 'UBottomLayer': UBottomLayer})
    model.load_weights(weight_path)

    filename_xte = 'sat_test_x_600.csv'
    raw_data_xte = open(filename_xte, 'rt')
    x_test_data = np.loadtxt(raw_data_xte, delimiter=",")
    x_test_data = x_test_data*100
    print(x_test_data.shape)

    x_predict = x_test_data[np.arange(0,1), :]
    outstep = []
    start_time = time.time()
    for stepi in np.arange(total_steps):
        out = model.predict(x_predict)
        x_predict = out
        outstep.append(out/100)
    print("--- %s seconds ---" % (time.time() - start_time))

    outstep = np.asarray(outstep)
    outstep = np.squeeze(outstep)
    print(outstep.shape)
    return outstep




trianing_saturation(True)
total_steps = 600
outs = prediction_saturation(total_steps)
filename_outstep = 'sat_pred_1phase.csv'
np.savetxt(filename_outstep, outs, delimiter=",")

