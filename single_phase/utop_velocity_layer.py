#!/usr/bin/python

import scipy.stats as stats
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import scipy.io
# from tensorflow.python.ops import sparse_ops
# from tensorflow.python.framework import sparse_tensor
# from tensorflow import sparse
# from tensorflow._api.v1.sparse import __init__

class UTopLayer(Layer):
  def __init__(self, output_dim, input_dim, **kwargs): #input_dim=None,
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(UTopLayer, self).__init__(**kwargs)


  def build(self, input_shape):
    mean = 0.0
    std = 0.01
    weight_init1 = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(12300,))
    self.W3 = K.variable(weight_init1)
    self.b = K.zeros((self.input_dim,))
    self.trainable_weights = [self.W3, self.b]

  def call(self, inputs, mask=None):
    weight_ind = scipy.io.loadmat('./data/w_ind.mat')
    wind = weight_ind['wind']
    I, J = wind.nonzero()
    I_n = I
    J_n = J

    input_sat = inputs
    file_vel = './data/velocity_Sat.csv'
    vel_raw = open(file_vel, 'rt')
    velocity = np.loadtxt(vel_raw, delimiter=",")
    velocity = np.asarray(velocity)
    velocity2 = 1250 * velocity[2, :]
    velocity = tf.constant(velocity2, dtype=tf.float32, shape=[self.input_dim, ])


    indices = [I_n, J_n]
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    ind = tf.constant(indices, dtype=tf.int64)
    sparse_w3 = tf.SparseTensor(indices=ind, values=self.W3, dense_shape=[self.input_dim, self.input_dim])

    sp_w3 = sparse_w3.__mul__(velocity)
    result = tf.transpose(tf.sparse_tensor_dense_matmul(sp_w3, tf.transpose(input_sat)))+ self.b
    return result

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0]
    return (batch_size, self.output_dim)

  def get_config(self):
    config = {'input_dim': self.input_dim, 'output_dim': self.output_dim}
    base_config = super(UTopLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))