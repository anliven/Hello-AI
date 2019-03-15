# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([5.2])  # 使用 tf.constant 指令定义常量
b = tf.Variable([5])  # 使用 tf.Variable 指令定义变量，必须指定默认值
b = b.assign([8])  # 分配新值
with tf.Session() as sess:  # 图必须在 TensorFlow 会话中运行，会话存储了它所运行的图的状态
    initialization = tf.global_variables_initializer()  # 使用 tf.Variable 时，必须先明确初始化变量
    print(b.eval())

g = tf.Graph()  # 虽然TensorFlow提供默认图， 但仍建议明确创建自己的Graph，以便跟踪状态

with g.as_default():  # 设置为默认图
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")  # tf.add 指令返回一个包含值之和的新张量
    z = tf.constant(4, name="z_const")
    new_sum = tf.add(sum, z, name="x_y_z_sum")
    with tf.Session() as sess:  # 创建会话，运行默认图
        print(new_sum.eval())

# ### 张量
# TensorFlow 的名称源自张量，张量是任意维度的数组。
# 借助 TensorFlow，可以操控具有大量维度的张量。但在大多数情况下，只会使用一个或多个低维张量：
#   - 标量是零维数组（零阶张量）。例如，\'Howdy\' 或 5
#   - 矢量是一维数组（一阶张量）。例如，[2, 3, 5, 7, 11] 或 [5]
#   - 矩阵是二维数组（二阶张量）。例如，[[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]
#
# ### 指令
# TensorFlow 指令会创建、销毁和操控张量。
# 典型 TensorFlow 程序中的大多数代码行都是指令。
#
# ### 图
# TensorFlow 图（也称为计算图或数据流图）是一种图数据结构。
# 很多 TensorFlow 程序由单个图构成，但是 TensorFlow 程序可以选择创建多个图。
# 图的节点是指令；图的边是张量。张量流经图，在每个节点由一个指令操控。
# 一个指令的输出张量通常会变成后续指令的输入张量。
# TensorFlow 会实现延迟执行模型，意味着系统仅会根据相关节点的需求在需要时计算节点。
#
# ### 张量与图
# 张量可以作为常量或变量存储在图中。
# 常量是始终会返回同一张量值的指令，存储的是值不会发生更改的张量。
# 变量是会返回分配给它的任何张量的指令，存储的是值会发生更改的张量。
#   - 张量：https://www.tensorflow.org/guide/tensors
#   - 变量：https://www.tensorflow.org/guide/variables
#
# ### 总结
# TensorFlow 编程本质上是一个两步流程：
#   1.将常量、变量和指令整合到一个图中。
#   2.在一个会话中评估这些常量、变量和指令。
