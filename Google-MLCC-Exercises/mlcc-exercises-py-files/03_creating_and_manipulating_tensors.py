# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

# 矢量（一维张量）加法
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)  # 包含质数的primes矢量
print("primes:", primes)
ones = tf.ones([6], dtype=tf.int32)  # 值全为1的ones矢量
print("ones:", ones)
just_beyond_primes = tf.add(primes, ones)  # 通过对前两个矢量执行元素级加法而创建的矢量
print("just_beyond_primes:", just_beyond_primes)
twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos  # 通过将primes矢量中的元素翻倍而创建的矢量
print("primes_doubled:", primes_doubled)
some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print("some_matrix: ", some_matrix)  # 输出张量将返回其值、形状以及存储在张量中的值的类型
print("\nvalue of some_matrix is:\n", some_matrix.numpy())  # 调用张量的numpy方法会返回该张量的值（以NumPy数组形式）

# 广播
primes2 = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
ones2 = tf.ones(1, dtype=tf.int32)  # 使用的是标量值（不是全包含1矢量）和广播
twos2 = tf.constant(2, dtype=tf.int32)  # 使用的是标量值（不是全包含 2 的矢量）和广播
just_beyond_primes2 = tf.add(primes2, ones2)
primes_doubled2 = primes2 * twos2
print("just_beyond_primes2:\n", just_beyond_primes2)
print("primes_doubled2:\n", primes_doubled2)

# 张量形状
scalar = tf.zeros([])  # 标量
vector = tf.zeros([3])  # 值全为0的矢量
matrix = tf.zeros([2, 3])  # 值全为0的2行3列矩阵
print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.numpy())
print('vector has shape', vector.get_shape(), 'and value:\n', vector.numpy())
print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.numpy())

# 练习
just_under_primes_squared = tf.subtract(tf.pow(primes, 2), ones)
print("just_under_primes_squared:", just_under_primes_squared)

# 矩阵乘法
x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype=tf.int32)  # 3行4列矩阵
y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)  # 4行2列矩阵
matrix_multiply_result = tf.matmul(x, y)  # 矩阵相乘的结果是3行2列矩阵
print("matrix_multiply_result:\n", matrix_multiply_result)

# 张量变形
matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.int32)  # 4行2列的矩阵
reshaped_2x4_matrix = tf.reshape(matrix, [2, 4])  # 将4x2张量变形为2x4张量
reshaped_1x8_matrix = tf.reshape(matrix, [1, 8])
reshaped_2x2x2_tensor = tf.reshape(matrix, [2, 2, 2])  # 将4x2张量变形为三维2x2x2张量
one_dimensional_vector = tf.reshape(matrix, [8])  # 将4x2张量变形为一维8元素张量
print("Original matrix (4x2):\n", matrix.numpy())
print("Reshaped matrix (2x4):\n", reshaped_2x4_matrix.numpy())
print("Reshaped matrix (1x8):\n", reshaped_1x8_matrix.numpy())
print("reshaped_2x2x2_tensor:\n", reshaped_2x2x2_tensor.numpy())
print("one_dimensional_vector:\n", one_dimensional_vector.numpy())

# 练习
a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 3])
c = tf.matmul(tf.reshape(a, [2, 3]), tf.reshape(b, [3, 1]))
print("a*b:\n", c.numpy())

# 变量、初始化和赋值
v = tf.contrib.eager.Variable([3])  # 创建一个初始值为3的标量变量
w = tf.contrib.eager.Variable(
    tf.random_normal(shape=[1, 4],  # 形状为1行4列，必选项
                     mean=1.0,  # 正态分布的均值，默认为0
                     stddev=0.35,  # 正态分布的标准差，默认为1.0
                     dtype=tf.float64,  # 输出的类型，默认为tf.float32
                     seed=1,  # 每次产生的随机数结果是否相同，如果固定seed值为一个整数则相同，默认为None（不相同）
                     name="test")  # 操作的名称
)  # 创建一个初始值为正态分布的1*4矢量变量
print("v:", v.numpy(), "w:", w.numpy())
tf.assign(v, [7])  # 使用assign更改变量的值
print("v:", v.numpy())
v.assign([5])
print("v:", v.numpy())
v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]])
print("v:", v.numpy())
try:
    print("Assigning [7, 8, 9] to v")
    v.assign([7, 8, 9])
except ValueError as e:
    print("Exception:", e)

# 练习
die1 = tf.contrib.eager.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
die2 = tf.contrib.eager.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
dice_sum = tf.add(die1, die2)
resulting_matrix = tf.concat(values=[die1, die2, dice_sum], axis=1)
print(resulting_matrix.numpy())

# ### 矢量加法
# 可以对张量执行很多典型数学运算：https://www.tensorflow.org/api_docs/python/tf/math；
# 输出张量将返回其值、形状以及存储在张量中的值的类型；
# 调用张量的numpy方法会返回该张量的值（以NumPy数组形式）；
#
# ### 广播
# TensorFlow支持广播（一种借鉴自NumPy的概念）；
# 利用广播，元素级运算中的较小数组会增大到与较大数组具有相同的形状；
#
# ### 张量形状（shape）
# 形状（shape）用于描述张量维度的大小和数量；
# 张量的形状表示为list，其中第i个元素表示维度i的大小；
# 列表的长度表示张量的阶（即维数）；
#
# ### 矩阵相乘
# 在线性代数中，当两个矩阵相乘时，第一个矩阵的列数必须等于第二个矩阵的行数，否则是无效的；
#
# ### 张量变形
# 可以使用tf.reshape方法改变张量的形状和维数（“阶”）；
# 例如，可以将4x2张量变形为2x4张量；
# 例如，可以将4x2张量变形为三维2x2x2张量或一维8元素张量；
#
# ### 变量、初始化和赋值
# 在TensorFlow中可以定义Variable对象(变量)，其值可以更改；
# 创建变量时，可以明确设置一个初始值，也可以使用初始化程序（例如分布）；
# 使用assign更改变量的值，向变量赋予新值时，其形状必须和之前的形状一致；
#
# ### tf.random_normal()函数
# 用于从服从指定正太分布的数值中取出指定个数的值；
