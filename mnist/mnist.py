# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 10:13
# @Author  : Equator

from keras.datasets import mnist
import matplotlib.pyplot as plt


def data_visual(X, y):
    fig = plt.figure()
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        # 子图尺寸自适应
        plt.tight_layout()
        print(i, y[i])
        plt.imshow(X[i], cmap='Greys')
        plt.title('Label:{}'.format(y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

data_visual(X_train, y_train)
