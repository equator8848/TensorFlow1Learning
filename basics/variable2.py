# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 10:58
# @Author  : Equator
import tensorflow as tf

W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.36), name='W')
b = tf.Variable(tf.zeros([4]), name='b')
print(W, b)

session = tf.Session()

# 使用saver恢复变量
saver = tf.train.Saver({'W': W, 'b': b})

# 只需要关心前缀即可
saver.restore(session, 'saver_test-0')

print(session.run(b))
