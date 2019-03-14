# coding=utf-8
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

tensor = tf.constant('Hello, world!')
tensor_value = tensor.numpy()
print(tensor_value)
