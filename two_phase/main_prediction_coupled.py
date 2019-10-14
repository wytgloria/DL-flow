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

    # return  K.mean(K.sqrt(energy_err/energy_true)*100)\
    #         +K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)/K.sum(K.square(y_true), axis=1))*100, axis=-1)\
    #         + K.mean(K.mean(K.abs(mass_err),axis=1), axis=-1)*0.25
    return  K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)/K.sum(K.square(y_true), axis=1))*100, axis=-1)\
             + K.mean(K.mean(K.abs(mass_err), axis=1), axis=-1) * 0.25



def compute_lmds(s):
    mu_w = 1
    mu_o = 5
    lambda_s = np.square(s)/ mu_w + np.square(s-1) / mu_o
    return lambda_s

def predForStep(total_step):
    # input data
    filename_xtr = './data/lamS53_test_single2.csv'
    raw_data_xtr = open(filename_xtr, 'rt')
    lmds_train = np.loadtxt(raw_data_xtr, delimiter=",")


    filename_str = './data/sat53_test_in2.csv'
    raw_data_str = open(filename_str, 'rt')
    sat_train = np.loadtxt(raw_data_str, delimiter=",")

    filename_utr = './data/usat53_test2.csv'
    raw_data_utr = open(filename_utr, 'rt')
    usat_test = np.loadtxt(raw_data_utr, delimiter=",")

    vel_weight_path = 'best_weights_velocity_l53_c.hdf5'
    vel_model_path = 'model_velocity_l53_c.h5'
    vel_model = load_model(vel_model_path, custom_objects={"mass_conserve_error": mass_conserve_error})
    vel_model.load_weights(vel_weight_path)

    sat_weight_path = 'best_weights_saturation_l53.hdf5'
    sat_model_path = 'model_saturation_l53.h5'
    sat_model = load_model(sat_model_path, custom_objects={'ULeftLayer': ULeftLayer, 'URightLayer': URightLayer,
                                                           'UTopLayer': UTopLayer, 'UBottomLayer': UBottomLayer})
    sat_model.load_weights(sat_weight_path)

    file_edge = './data/EdgeLoc.csv'
    edge_raw = open(file_edge, 'rt')
    egdeloc = np.loadtxt(edge_raw, delimiter=",")
    egdeloc = np.asarray(egdeloc, dtype=np.int32)

    print('start prediction')
    img_rows, img_cols = int(np.sqrt(x_train.shape[1])), int(np.sqrt(x_train.shape[1]))
    lmds_train = lmds_train.reshape((1,img_rows, img_cols, 1))
    velocity = vel_model.predict(lmds_train)/100

    predrange0 = np.arange(0, 1)
    sat_in = sat_train[predrange0, :]
    sat_in = sat_in * 100
    outsat = []
    outvel = []

    start_time = time.time()
    for casei in np.arange(total_step):
        velocity = velocity * 100

        u_test1 = 50 * velocity[:, egdeloc[0, :]]
        u_test2 = -50 * velocity[:, egdeloc[1, :]]
        u_test3 = 50 * velocity[:, egdeloc[2, :]]
        u_test4 = -50 * velocity[:, egdeloc[3, :]]

        ###### then predict next step saturation using current sat and velocity
        sat_out = sat_model.predict([sat_in, u_test1, u_test2, u_test3, u_test4])
        ###### calculate lambda_s
        lambda_s = compute_lmds(sat_out/100)

        ######  update velocity and saturation
        #use predicted velocity
        lambda_s = lambda_s.reshape((1,img_rows, img_cols, 1))
        velocity = vel_model.predict(lambda_s) / 10000
        #use true velocity
        # predrange = np.arange(casei, casei + 1)
        # velocity = usat_test[predrange, :]

        sat_in = sat_out
        outsat.append(sat_out/100)
        outvel.append(velocity)

    print("--- %s seconds ---" % (time.time() - start_time))
    outsat = np.asarray(outsat)
    outsat = np.squeeze(outsat)
    outvel = np.asarray(outvel)
    outvel = np.squeeze(outvel)
    return outsat, outvel


# main command
total_step = 199
outsat, outvel = predForStep(total_step)
filename_outsat = 'sat_coupled_pred_l53.csv'
np.savetxt(filename_outsat, (outsat), delimiter=",")

