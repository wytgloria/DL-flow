'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

# source ~/tensorflow/bin/activate
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import numpy as np
import h5py
import keras
import pickle
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
from uleft_velocity_layer import ULeftLayer
from uright_velocity_layer import URightLayer
from utop_velocity_layer import UTopLayer
from ubottom_velocity_layer import UBottomLayer
from keras.layers.merge import add
from keras.constraints import Constraint
from keras.initializers import Initializer
import scipy.io
import tensorflow as tf
from tensorflow.python.ops import sparse_ops as sparse
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b, permutation

def trianing_saturation(sparse = True):
    # input data
    filename_xtr = './data/sat53_train_in2.csv'
    raw_data_xtr = open(filename_xtr, 'rt')
    x_raw_data = np.loadtxt(raw_data_xtr, delimiter=",")#, usecols=range(10081))

    filename_utr = './data/usat53_train2.csv'
    raw_data_utr = open(filename_utr, 'rt')
    u_raw_data = np.loadtxt(raw_data_utr, delimiter=",")  # , usecols=range(10081))

    filename_ytr = './data/sat53_train_out2.csv'
    raw_data_ytr = open(filename_ytr, 'rt')
    y_raw_data = np.loadtxt(raw_data_ytr, delimiter=",")#, usecols=range(81))

    x_train = x_raw_data*100
    u_train = u_raw_data*100
    y_train = y_raw_data*100
    print(x_train.shape, 'train ins samples')
    print(u_train.shape, 'train inu samples')
    print(y_train.shape, 'train outs samples')

    file_edge = './data/EdgeLoc.csv'
    edge_raw = open(file_edge, 'rt')
    egdeloc = np.loadtxt(edge_raw, delimiter=",")
    egdeloc = np.asarray(egdeloc, dtype=np.int32)

    u_train1 =  50*u_train[:, egdeloc[0, :]]
    u_train2 = -50*u_train[:, egdeloc[1, :]]
    u_train3 = 50*u_train[:, egdeloc[2, :]]
    u_train4 = -50*u_train[:, egdeloc[3, :]]

    epochs = 500
    batch_size = 10
    alpha=0.3

    input_dim = x_train.shape[1]
    out_dim = input_dim
    input1 = Input(shape=(input_dim,))
    inputu1 = Input(shape=(input_dim,))
    inputu2 = Input(shape=(input_dim,))
    inputu3 = Input(shape=(input_dim,))
    inputu4 = Input(shape=(input_dim,))

    out1 = ULeftLayer(input_dim=input_dim, output_dim=out_dim)([input1, inputu1])
    out1 = Activation('relu')(out1)
    out2 = URightLayer(input_dim=input_dim, output_dim=out_dim)([input1, inputu2])
    out2 = Activation('relu')(out2)
    out3 = UTopLayer(input_dim=input_dim, output_dim=out_dim)([input1, inputu3])
    out3 = Activation('relu')(out3)
    out4 = UBottomLayer(input_dim=input_dim, output_dim=out_dim)([input1, inputu4])
    out4 = Activation('relu')(out4)

    sum1 = add([out1, out2])
    sum2 = add([out3, sum1])
    sum3 = add([out4, sum2])

    out5 = Dense(50)(sum3)
    out5 = LeakyReLU(alpha)(out5)
    out6 = Dense(out_dim)(out5)
    out6 = LeakyReLU(alpha)(out6)

    sum4 = add([out6, input1])
    output = sum4

    model = Model([input1, inputu1, inputu2, inputu3, inputu4], output)
    model.summary()

    admax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    weight_path = 'best_weights_saturation_l53.hdf5'
    checkpointer = ModelCheckpoint(filepath=weight_path, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    model.compile(loss= mse,
                  optimizer= admax,
                  metrics=[mse])

    history = model.fit([x_train, u_train1, u_train2, u_train3, u_train4] , y_train,
                        epochs=epochs,
                        batch_size = batch_size,
                        verbose=1,
                        callbacks = [checkpointer]) #validation_data=(x_test, y_test)
    model_path = 'model_saturation_l53.h5'
    model.save(model_path)


def prediction_saturation(total_steps):
    weight_path = 'best_weights_saturation_l53.hdf5'
    model_path = 'model_saturation_l53.h5'

    model = load_model(model_path, custom_objects={'ULeftLayer': ULeftLayer, 'URightLayer':URightLayer,
                                                   'UTopLayer':UTopLayer, 'UBottomLayer':UBottomLayer})
    model.load_weights(weight_path)

    admax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=admax, metrics=['mse'])

    filename_xte = './data/sat53_test_in2.csv'
    raw_data_xte = open(filename_xte, 'rt')
    x_test_data = np.loadtxt(raw_data_xte, delimiter=",")
    print(x_test_data.shape)
    x_test_data = x_test_data*100


    filename_utr = './data/usat53_test2.csv'
    raw_data_utr = open(filename_utr, 'rt')
    u_test_data = np.loadtxt(raw_data_utr, delimiter=",")
    u_test_data = u_test_data*100

    file_edge = './data/EdgeLoc.csv'
    edge_raw = open(file_edge, 'rt')
    egdeloc = np.loadtxt(edge_raw, delimiter=",")
    egdeloc = np.asarray(egdeloc, dtype=np.int32)

    u_test1 = 50 * u_test_data[:, egdeloc[0, :]]
    u_test2 = -50 * u_test_data[:, egdeloc[1, :]]
    u_test3 = 50 * u_test_data[:, egdeloc[2, :]]
    u_test4 = -50 * u_test_data[:, egdeloc[3, :]]

    outs = []
    for casei in np.arange(total_steps):
        predrange0 = np.arange(casei, casei + 1)
        x_predict = x_test_data[predrange0, :]
        u_predict1 = u_test1[predrange0, :]
        u_predict2 = u_test2[predrange0, :]
        u_predict3 = u_test3[predrange0, :]
        u_predict4 = u_test4[predrange0, :]

        out = model.predict([x_predict, u_predict1, u_predict2, u_predict3, u_predict4])
        out = out/100
        outs.append(out)

    outs = np.asarray(outs)
    outs = np.squeeze(outs)
    print(outs.shape)
    return outs



# main command
trianing_saturation(True)
total_steps = 199
prediction_saturation(total_steps)
filename_out = 'sat_only_pred_l53_batch10.csv'
np.savetxt(filename_out, (out_alltest), delimiter=",")

