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

save_path = './model/'
model_name = 'softmax.h5'
model_path = os.path.join(save_path, model_name)
model = load_model(model_path)

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将图像从(28,28)转换为(784,1)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 归一化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# one-hot编码
n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

loss_and_metrics = model.evaluate(x_test, y_test, verbose=2)
print('Test Loss: {}'.format(loss_and_metrics[0][0]))
print('Test Accuracy: {}%'.format(loss_and_metrics[0][1] * 100))
predicted_classes = model.predict_classes(x_test)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print('Classified correctly count: {}'.format(len(correct_indices)))
print('Classified incorrectly count: {}'.format(len(incorrect_indices)))
