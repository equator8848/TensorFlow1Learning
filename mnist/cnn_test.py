# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 19:29
# @Author  : Equator

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
import os
import numpy as np
import tensorflow.gfile as gfile
from keras import backend as K

save_path = './model/'
model_name = 'cnn.h5'
model_path = os.path.join(save_path, model_name)
model = load_model(model_path)

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 通道格式设置
img_rows, img_cols = 28, 28,
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols),
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(x_train.shape, type(x_train)),
print(x_test.shape, type(x_test))

# 将数据类型转换为float32
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
# 数据归一化
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one-hot编码
n_classes = 10
print('Shape before one-hot encoding: ', y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
print('Shape after one-hot encoding:', Y_train.shape)
Y_test = np_utils.to_categorical(y_test, n_classes)

loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
print('Test Loss: {}'.format(loss_and_metrics[0]))
print('Test Accuracy: {}%'.format(loss_and_metrics[1] * 100))
predicted_classes = model.predict_classes(x_test)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print('Classified correctly count: {}'.format(len(correct_indices)))
print('Classified incorrectly count: {}'.format(len(incorrect_indices)))
