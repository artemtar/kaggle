#!/usr/bin/python3.6
import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import numpy as np
import tensorflow as tf


#data for project is taken from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

TRAIN_DIR = '/home/artem/Documents/dogs_vs_cats/train'
TEST_DIR = '/home/artem/Documents/dogs_vs_cats/test'
SAVE_DIR = '/home/artem/Documents/dogs_vs_cats/save'
IMG_SIZE = 50
LR = 1e-3
BATCH = 100

MODEL_NAME = 'DogVsCats-learningRate {}, model {}'.format(LR, 'conv2D')


def get_tag(img):
    lable = img.split('.')[-3]
    if lable == 'cat': return [1, 0]
    if lable == 'dog': return [0, 1]


def creating_train_data_set():
    data_set = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        lable = get_tag(img)
        path = os.path.join(TRAIN_DIR, img)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data_set.append([np.array(image), np.array(lable)])
    random.shuffle(data_set)
    np.save(os.path.join(SAVE_DIR, 'train_data.npy'), data_set)

def creating_test_data_set():
    data_set = []
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num = img.split('.')[0]
        path = os.path.join(TEST_DIR, img)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data_set.append([np.array(image), np.array(img_num)])
    random.shuffle(data_set)
    np.save(os.path.join(SAVE_DIR, 'test_data.npy'), data_set)


if not os.path.exists(os.path.join(SAVE_DIR, 'train_data.npy')):
    creating_train_data_set()
if not os.path.exists(os.path.join(SAVE_DIR, 'test_data.npy')):
    creating_test_data_set()

data_train = np.load(os.path.join(SAVE_DIR, 'train_data.npy'))
data_test = np.load(os.path.join(SAVE_DIR, 'test_data.npy'))

#model

input_x = tf.placeholder(dtype='float32', shape=(None, IMG_SIZE, IMG_SIZE), name='input')
lable_y = tf.placeholder(dtype='int32', shape=(None, 2), name='lable')

input_x = tf.reshape(input_x, shape=(-1, IMG_SIZE, IMG_SIZE, 1))
lable_y = tf.reshape(lable_y, shape=(-1, 2))
conv1 = tf.layers.conv2d(inputs=input_x,
                         filters=16,
                         kernel_size=[5,5],
                         strides=[1,1],
                         padding='same',
                         activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2,2],
                                strides=[2,2],
                                padding='same')

conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=32,
                         kernel_size=[3,3],
                         strides=[1,1],
                         padding='same',
                         activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[3,3],
                                strides=[3,3],
                                padding='same')

flatten = tf.layers.flatten(inputs=pool2)

fully_connected = tf.layers.dense(inputs=flatten,
                                  activation=tf.nn.relu,
                                  units=256)
drop = tf.layers.dropout(fully_connected, rate=0.6)

logits = tf.layers.dense(inputs=drop,
                         units=2)

class data_iter:
    def __init__(self, iterable, batch):
        self.start = 0
        self.container = iterable
        self.batch_size = batch
    def get_next(self):
        start = self.start
        next_bound = self.start + self.batch_size
        if next_bound > len(self.container):
            return False
        else:
            self.start = next_bound
            return self.container[start:next_bound]
# data iterator



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lable_y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

d = data_iter(data_train, BATCH)

def train(epoch = 1):

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for e in range(epoch):
            data = d.get_next()
            iter = 0
            loss_val = 0
            while(data != False):
                y = [y[1] for y in data]
                x = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                dict = {input_x:x, lable_y:y}
                l, _, c_e = sess.run([loss, optimizer, cross_entropy], feed_dict=dict)
                accuracy(y, c_e)
                data = d.get_next()
                loss_val += l
            print('epoch number {}, loss value {}'.format(e + 1, loss_val))

def accuracy (pred, correct):
    y_pred = pred
    y_correct = correct
    correct = tf. argmax(y_correct)
    pred = tf.argmax(y_pred)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run([correct], feed_dict={})

train()