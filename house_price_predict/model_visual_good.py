# -*- coding: utf-8 -*-
# @Time    : 2021/3/5 19:25
# @Author  : Equator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import tensorflow as tf

df = pd.read_csv('data_multi.csv', names=['square', 'bedrooms', 'price'])
print(df.columns)
df = df.apply(lambda col: (col - col.mean()) / col.std())
ones = pd.DataFrame({'ones': np.ones(len(df))})
# 根据列合并数据
df = pd.concat([ones, df], axis=1)

X_data = np.array((df[df.columns[0:3]]))
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)

print(X_data.shape, type(X_data))
print(y_data.shape, type(y_data))

# 学习率
alpha = 0.01
# 训练全量数据集的轮数
epoch = 500

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, X_data.shape)
    y = tf.placeholder(tf.float32, y_data.shape)

with tf.name_scope('hypothesis'):
    W = tf.get_variable('weights', (X_data.shape[1], 1), initializer=tf.constant_initializer())
    # 假设函数 h(x) = w0*x0 + w1*x1 + w2*x2，其中x0恒为1
    y_predict = tf.matmul(X, W)

with tf.name_scope('loss'):
    # 损失值采用最小二乘法 先转置再相乘
    loss_op = 1 / (2 * len(X_data)) * tf.matmul(y_predict - y, y_predict - y, transpose_a=True)

with tf.name_scope('train'):
    # 随机梯度下降优化器
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    # 单轮训练
    train_op = opt.minimize(loss_op)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # 创建fileWriter实例
    writer = tf.summary.FileWriter('./summary/house-price-predict-good-visual/', session.graph)
    for e in range(1, epoch + 1):
        session.run(train_op, feed_dict={X: X_data, y: y_data})
        if e % 2 == 0:
            loss, w = session.run([loss_op, W], feed_dict={X: X_data, y: y_data})
            print("Epoch %d \t Loss=%.4g \t Model: y = %.4gx1+%.4gx2+%.4g" % (e, loss, w[1], w[2], w[0]))
writer.close()
