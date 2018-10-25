#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:42:06 2018

@author: zengyang
"""

## read data
import numpy as np
import os
import tensorflow as tf

tf.reset_default_graph()

## classes for discriminating
classes = {'seastate_ss1','images_ss1'}

#### read data from the path
cwd = '/Users/zengyang/research/Final_Wake_Files/'

### input data
data_input = []
data_label = []

for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    label = np.zeros([2])
    label[index] = 1
    for data_name in os.listdir(class_path):
        if data_name == '.DS_Store':
            continue
        data_path = class_path + data_name
        data = np.loadtxt(data_path)
        data_input.append(data)
        data_label.append(label)
data_input = np.asarray(data_input)
data_label = np.asarray(data_label)


# random the data
def next_batch(num, labels, U):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(labels))
    np.random.shuffle(idx)
    idx = idx[:num]
    
    U_shuffle = [U[i] for i in idx]
    label_shuffle = [labels[i] for i in idx]
    return np.asarray(U_shuffle), np.asarray(label_shuffle)

# activation function
def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))       


# discriminator for the images
def discriminator(data, isTrain=True, reuse=False):
    keep_prob = 0.6
    activation = lrelu
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.reshape(data, shape=[-1,100,100,1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=32, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.batch_normalization(x,training=isTrain)
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.batch_normalization(x,training=isTrain)
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, 2, [7, 7], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(x)
        return o
# place holder for the input of NN
NN_input = tf.placeholder(dtype=tf.float32, shape=[None, 100, 100], name='NN_input')
NN_label = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='NN_label')

# place holder for the accuracy computation
NN_label_cls = tf.argmax(NN_label, axis=1)
isTrain = tf.placeholder(dtype=tf.bool)

# discriminate
pre_label = discriminator(NN_input, isTrain)
pre_label = tf.reshape(pre_label, shape=[-1,2])
pre_label_cls = tf.argmax(pre_label, axis=1)
correct_prediction = tf.equal(pre_label_cls, NN_label_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# variable for trainging
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)

# loss function cross entropy
loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=NN_label, logits=pre_label)
loss = tf.reduce_mean(loss1)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# optimize
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss, var_list=vars_d)

# batch_size for training
batch_size = 32


data_input_rand, data_label_rand = next_batch(data_input.shape[0], data_label, data_input)

# number of training
num_training = int(data_input_rand.shape[0]*0.7)

# training data
training_input = data_input_rand[0:num_training]
training_label = data_label_rand[0:num_training]

# test data
test_input = data_input_rand[num_training:-1]
test_label = data_label_rand[num_training:-1]

train_epoch = 30
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(train_epoch):
    for iter in range(int(training_input.shape[0] // batch_size)):
        batch = training_input[iter*batch_size:(iter+1)*batch_size]
        label = training_label[iter*batch_size:(iter+1)*batch_size]
        loss_,_ = sess.run([loss, optimizer_d], feed_dict={NN_input: batch, NN_label: label, isTrain:True})
    acc_train = sess.run(accuracy, feed_dict={NN_input:training_input, NN_label: training_label, isTrain:False})    
    acc_test = sess.run(accuracy, feed_dict={NN_input:test_input, NN_label: test_label, isTrain:False})
    msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
    print(msg.format(i + 1, acc_train))
    msg = "Optimization Iteration: {0:>6}, Test Accuracy: {1:>6.1%}"
    print(msg.format(i + 1, acc_test))
    





