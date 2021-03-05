# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 10:58
# @Author  : Equator
import tensorflow as tf

W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.36), name='W')
b = tf.Variable(tf.zeros([4]), name='b')
print(W, b)

session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run([W, b]))

print(session.run(tf.assign_add(b, [1, 1, 1, 1])))
print(session.run(b))

# 使用saver保存变量
saver = tf.train.Saver({'W': W, 'b': b})
# 另一种方式，接受一个数组 saver = tf.train.Saver([W, b])
# global_step 指明训练到哪一步了
saver.save(session, 'saver_test', global_step=0)
