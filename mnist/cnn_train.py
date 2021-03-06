# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 20:28
# @Author  : Equator

from keras.datasets import mnist
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import os
import tensorflow.gfile as gfile

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, type(x_train))
print(y_train.shape, type(y_train))

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

model = Sequential()
# Feature Extraction
# 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape)),

# 第2层卷积，64个3x3的卷积核，激活函数使用 relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 最大池化层，池化窗口 2x2,
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout 25% 的输入神经元
model.add(Dropout(0.25))

# 将 Pooled feature map 摊平后输入全连接网络
model.add(Flatten())

# Classification
# 全联接层,
model.add(Dense(128, activation='relu'))

# Dropout 50% 的输入神经元
model.add(Dropout(0.5))

# 使用 softmax 激活函数做多分类，输出各数字的概率
model.add(Dense(n_classes, activation='softmax'))

print(model.summary())

# 编译模型
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# 训练模型
history = model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=2, validation_data=(X_test, Y_test))

# 数据可视化
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.show()

# 保存模型
save_path = './model/'

if gfile.Exists(save_path):
    gfile.DeleteRecursively(save_path)
gfile.MakeDirs(save_path)

model_name = 'cnn.h5'
model_path = os.path.join(save_path, model_name)
model.save(model_path)
