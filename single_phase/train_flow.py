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
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, LocallyConnected2D
from keras.losses import mse
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Input, Concatenate, Lambda, Activation
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import scipy.io
from tensorflow.python.ops import sparse_ops as sparse
import time


def mass_conserve_error(y_true, y_pred):
    B_mat_l = scipy.io.loadmat('./data/B.mat')
    B_mat = B_mat_l['B']
    dataB = tf.cast(B_mat.data, tf.float32)
    Ib, Jb = B_mat.nonzero()
    indicesb = [Ib, Jb]
    indicesb = np.asarray(indicesb)
    indicesb = np.transpose(indicesb)
    indb = tf.constant(indicesb, dtype=tf.int64)
    B_mat = tf.SparseTensor(indices=indb, values=dataB, dense_shape=[2500, 5100])
    mass_err = tf.transpose(tf.sparse_tensor_dense_matmul(B_mat, tf.transpose(y_true-y_pred)))
    mass_true = tf.transpose(tf.sparse_tensor_dense_matmul(B_mat, tf.transpose(y_true)))

    AM_mat_a = scipy.io.loadmat('./data/AM_mat_l53.mat')
    AM_mat = AM_mat_a['AM']
    data =  tf.cast(AM_mat.data, tf.float32)
    I, J = AM_mat.nonzero()
    indices = [I, J]
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    ind = tf.constant(indices, dtype=tf.int64)
    AM_mat = tf.SparseTensor(indices=ind, values=data, dense_shape=[5100,5100])

    energy_err = tf.diag_part(tf.matmul(y_pred - y_true, tf.sparse_tensor_dense_matmul(AM_mat, tf.transpose(y_pred - y_true))))
    energy_true =tf.diag_part(tf.matmul(y_true, tf.sparse_tensor_dense_matmul(AM_mat, tf.transpose(y_true))))
    energy_pred = tf.diag_part(tf.matmul(y_pred, tf.sparse_tensor_dense_matmul(AM_mat, tf.transpose(y_pred))))

    return  K.mean(K.sqrt(energy_err/energy_true)*100)\
            +K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)/K.sum(K.square(y_true), axis=1))*100, axis=-1)\
            + K.mean(K.mean(K.abs(mass_err),axis=1), axis=-1)*0.25


def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b, permutation

def train_model(lcn = True, custom_loss = True):
    # input data
    filename_xtr = './data/source_train.csv'
    raw_data_xtr = open(filename_xtr, 'rt')
    x_raw_data = np.loadtxt(raw_data_xtr, delimiter=",")

    filename_ytr = './data/velocityF_train.csv'
    raw_data_ytr = open(filename_ytr, 'rt')
    y_raw_data = np.loadtxt(raw_data_ytr, delimiter=",")

    # x_raw_data, y_raw_data, permutation = randomize(x_raw_data, y_raw_data)

    x_train = x_raw_data
    y_train = y_raw_data*10000

    img_rows, img_cols = int(np.sqrt(x_train.shape[1])), int(np.sqrt(x_train.shape[1]))
    x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, 1))
    input_shape = (img_rows, img_cols, 1)

    print(x_train.shape, 'train ins samples')
    print(y_train.shape, 'train outs samples')

    epochs = 250
    batch_size = 100
    alpha=0.3
    coarse_grid = 10
    pool_window = int(img_rows/coarse_grid)
    model = Sequential()

    if lcn:
        model.add(AveragePooling2D(pool_size=(pool_window, pool_window), input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(np.square(coarse_grid)))
        model.add(LeakyReLU(alpha))
        model.add(Reshape((coarse_grid, coarse_grid, 1)))
        model.add(LocallyConnected2D(4, kernel_size=(3, 3)))  # strides = (3,3), padding='same'
        model.add(LeakyReLU(alpha))
        model.add(LocallyConnected2D(8, kernel_size=(3, 3)))  # strides=(3, 3)
        model.add(LeakyReLU(alpha))
        model.add(Flatten())

    else:
        model.add(AveragePooling2D(pool_size=(pool_window, pool_window), input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(np.square(coarse_dof)))
        model.add(LeakyReLU(alpha))
        model.add(Reshape((coarse_dof, coarse_dof, 1)))
        model.add(Conv2D(4, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha))
        model.add(Conv2D(8, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha))
        model.add(Flatten())


    model.add(Dense(y_train.shape[1]))
    model.add(LeakyReLU(alpha))

    model.summary()

    admax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if lcn:
        weight_path = 'best_weights_conserv_lcn.hdf5'
    else:
        weight_path = 'best_weights_conserv_cnn.hdf5'

    checkpointer = ModelCheckpoint(filepath=weight_path, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    if custom_loss:
        loss_function = mass_conserve_error
    else:
        loss_function = mse

    model.compile(loss= loss_function,
                  optimizer= admax,
                  metrics=[mse])


    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size = batch_size,
                        verbose=1,
                        callbacks = [checkpointer])
    if lcn:
        model_path = 'model_convserv_lcn.h5'
    else:
        model_path = 'model_convserv_cnn.h5'

    model.save(model_path)

def prediction(lcn):
    if lcn:
        weight_path = 'best_weights_conserv_lcn.hdf5'
    else:
        weight_path = 'best_weights_conserv_cnn.hdf5'

    if lcn:
        model_path = 'model_convserv_lcn.h5'
    else:
        model_path = 'model_convserv_cnn.h5'

    if custom_loss:
        model = load_model(model_path, custom_objects={"mass_conserve_error": mass_conserve_error})
    else:
        model = load_model(model_path)


    model.load_weights(weight_path)

    admax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=admax, metrics=['mse'])

    filename_xte =  './data/source_test.csv'
    raw_data_xte = open(filename_xte, 'rt')
    x_test_data = np.loadtxt(raw_data_xte, delimiter=",")

    filename_yte = './data/velocityF_test.csv'
    raw_data_yte = open(filename_yte, 'rt')
    y_test_data = np.loadtxt(raw_data_yte, delimiter=",")
    print(y_test_data.shape)

    x_test_data = x_test_data
    y_test_data = y_test_data*10000

    errors = []
    outs = []
    for casei in np.arange(250):
        predrange0 = np.arange(casei, casei + 1)
        x_predict = x_test_data[predrange0, :]
        y_predict = y_test_data[predrange0]
        start_time = time.time()
        out = model.predict(x_predict) / 10000
        print("--- %s seconds ---" % (time.time() - start_time))

        outs.append(out)

    outs = np.asarray(outs)
    # errors = np.squeeze(errors)
    outs = np.squeeze(outs)
    print(outs.shape)
    return outs


lcn = True
custom_loss = True
training = True
if training:
    train_model(lcn, custom_loss)
else:
    out_alltest = prediction(lcn, custom_loss)
    filename_out = 'OutF_l53_dnn.csv'
    np.savetxt(filename_out, out_alltest, delimiter=",")
