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

input = tf.placeholder(dtype='float32', shape=[None, IMG_SIZE, IMG_SIZE, 1], name='name')

conv1 = tf.layers.conv2d(inputs=input,
                         filters=16,
                         kernel_size=[5,5],
                         padding='same',
                         activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2,2],
                                strides=[2,2],
                                padding='same')

