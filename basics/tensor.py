# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 10:37
# @Author  : Equator

import tensorflow as tf

# 零阶张量 （标量）
zero = tf.Variable(98, tf.int32)
print(zero)

# 一阶张量 （向量）
one = tf.Variable(['hello', 'world'], tf.string, name='Hello')
print(one)

# 二阶张量 （矩阵）
two = tf.Variable([[True, False], [False, True]], tf.bool)
print(two)

rank = tf.rank(two)
print(rank)

# 三阶张量 （立方体）
three = tf.zeros([6, 7, 8])
print(three)
