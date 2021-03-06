# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 18:52
# @Author  : Equator

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import tensorflow.gfile as gfile

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

# 模型定义
model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

# 模型编译
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# 模型训练，将指标保存到history
history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=2, validation_data=(x_test, y_test))

# 数据可视化
fig = plt.figure(),
plt.subplot(2, 1, 1),
plt.plot(history.history['acc']),
plt.plot(history.history['val_acc']),
plt.title('Model Accuracy'),
plt.ylabel('accuracy'),
plt.xlabel('epoch'),
plt.legend(['train', 'test'], loc='lower right'),
plt.subplot(2, 1, 2),
plt.plot(history.history['loss']),
plt.plot(history.history['val_loss']),
plt.title('Model Loss'),
plt.ylabel('loss'),
plt.xlabel('epoch'),
plt.legend(['train', 'test'], loc='upper right'),
plt.tight_layout(),
plt.show()

# 保存模型
save_path = './model/'

if gfile.Exists(save_path):
    gfile.DeleteRecursively(save_path)
gfile.MakeDirs(save_path)

model_name = 'softmax.h5'
model_path = os.path.join(save_path, model_name)
model.save(model_path)
