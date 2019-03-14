# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g = tf.Graph()

with g.as_default():
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")
    z = tf.constant(4, name="z_const")
    new_sum = tf.add(sum, z, name="x_y_z_sum")
    with tf.Session() as sess:
        print(new_sum.eval())
