# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 0:00
# @Author  : Equator

import tensorflow as tf
# 定义常量操作
say_hello = tf.constant("Hello TensorFlow !")
# 创建会话
session = tf.Session()
# 运行会话
print(session.run(say_hello))
