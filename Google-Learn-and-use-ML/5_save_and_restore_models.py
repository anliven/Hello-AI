# coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("# TensorFlow version: {}  - tf.keras version: {}".format(tf.VERSION, tf.keras.__version__))  # 查看版本

# ### 获取示例数据集
ds_path = str(pathlib.Path.cwd()) + "\\datasets\\mnist\\"  # 数据集路径
# 查看numpy格式数据
np_data = np.load(ds_path + "mnist.npz")
print("np_data keys: ", list(np_data.keys()))  # 查看所有的键
# 加载mnist数据集
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data(path=ds_path + "mnist.npz")
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# ### 定义模型
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])  # 构建一个简单的模型
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


mod = create_model()
mod.summary()

# ### 保存和恢复模型（Save and restore models）
# 官网示例：https://www.tensorflow.org/tutorials/keras/save_and_restore_models
#
# ### MNIST数据集
# MNIST（Mixed National Institute of Standards and Technology database）是一个计算机视觉数据集
# - 官方下载地址：http://yann.lecun.com/exdb/mnist/
# - 包含70000张手写数字的灰度图片，其中60000张为训练图像和10000张为测试图像
# - 每一张图片都是28*28个像素点大小的灰度图像
# - https://keras.io/datasets/#mnist-database-of-handwritten-digits
# - TensorFlow：https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
