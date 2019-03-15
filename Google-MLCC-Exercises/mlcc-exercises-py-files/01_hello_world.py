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

# ### Eager Execution
# https://www.tensorflow.org/guide/eager?hl=zh-cn
# https://www.tensorflow.org/api_docs/python/tf/enable_eager_execution
# TensorFlow的“Eager Execution”是一个命令式、由运行定义的接口，一旦从 Python 被调用可立即执行操作；
#   - 操作会返回具体的值，而不是构建以后再运行的计算图；
#   - 能够轻松地开始使用TensorFlow和调试模型，并且还减少了样板代码；
#   - 使得 TensorFlow 的入门变得更简单，也使得研发工作变得更直观；
