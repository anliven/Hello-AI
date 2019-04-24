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
np_data = np.load(ds_path + "mnist.npz")  # 加载numpy格式数据
print("# np_data keys: ", list(np_data.keys()))  # 查看所有的键

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

# ### 在训练期间保存检查点

# 检查点回调用法
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)  # 检查点存放目录
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2)  # 创建检查点回调
model1 = create_model()
model1.fit(train_images, train_labels,
           epochs=10,
           validation_data=(test_images, test_labels),
           verbose=0,
           callbacks=[cp_callback]  # 将ModelCheckpoint回调传递给该模型
           )  # 训练模型，将创建一个TensorFlow检查点文件集合，这些文件在每个周期结束时更新

model2 = create_model()  # 创建一个未经训练的全新模型（与原始模型架构相同，才能分享权重）
loss, acc = model2.evaluate(test_images, test_labels)  # 使用测试集进行评估
print("# Untrained model2, accuracy: {:5.2f}%".format(100 * acc))  # 未训练模型的表现（准确率约为10%）

model2.load_weights(checkpoint_path)  # 从检查点加载权重
loss, acc = model2.evaluate(test_images, test_labels)  # 使用测试集，重新进行评估
print("# Restored model2, accuracy: {:5.2f}%".format(100 * acc))  # 模型表现得到大幅提升

# 检查点回调选项
checkpoint_path2 = "training_2/cp-{epoch:04d}.ckpt"  # 使用“str.format”方式为每个检查点设置唯一名称
checkpoint_dir2 = os.path.dirname(checkpoint_path)
cp_callback2 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path2,
                                                  verbose=1,
                                                  save_weights_only=True,
                                                  period=5  # 每隔5个周期保存一次检查点
                                                  )  # 创建检查点回调
model3 = create_model()
model3.fit(train_images, train_labels,
           epochs=50,
           callbacks=[cp_callback2],  # 将ModelCheckpoint回调传递给该模型
           validation_data=(test_images, test_labels),
           verbose=0)  # 训练一个新模型，每隔5个周期保存一次检查点并设置唯一名称
latest = tf.train.latest_checkpoint(checkpoint_dir2)
print("# latest checkpoint: {}".format(latest))  # 查看最新的检查点

model4 = create_model()  # 重新创建一个全新的模型
loss, acc = model2.evaluate(test_images, test_labels)  # 使用测试集进行评估
print("# Untrained model4, accuracy: {:5.2f}%".format(100 * acc))  # 未训练模型的表现（准确率约为10%）

model4.load_weights(latest)  # 加载最新的检查点
loss, acc = model4.evaluate(test_images, test_labels)  #
print("# Restored model4, accuracy: {:5.2f}%".format(100 * acc))  # 模型表现得到大幅提升

# ### 手动保存权重
model5 = create_model()
model5.fit(train_images, train_labels,
           epochs=10,
           validation_data=(test_images, test_labels),
           verbose=0)  # 训练模型
model5.save_weights('./training_3/my_checkpoint')  # 手动保存权重

model6 = create_model()
loss, acc = model6.evaluate(test_images, test_labels)
print("# Restored model6, accuracy: {:5.2f}%".format(100 * acc))
model6.load_weights('./training_3/my_checkpoint')
loss, acc = model6.evaluate(test_images, test_labels)
print("# Restored model6, accuracy: {:5.2f}%".format(100 * acc))

# ### 保存整个模型
model7 = create_model()
model7.fit(train_images, train_labels, epochs=5)
model7.save('my_model.h5')  # 保存整个模型到HDF5文件

model8 = keras.models.load_model('my_model.h5')  # 重建完全一样的模型，包括权重和优化器
model8.summary()
loss, acc = model8.evaluate(test_images, test_labels)
print("Restored model8, accuracy: {:5.2f}%".format(100 * acc))

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
#
# ### 在训练期间保存检查点
# 在训练期间或训练结束时自动保存检查点。
# 权重存储在检查点格式的文件集合中，这些文件仅包含经过训练的权重（采用二进制格式）。
# 可以使用经过训练的模型，而无需重新训练该模型，或从上次暂停的地方继续训练，以防训练过程中断
# - 检查点回调用法：创建检查点回调，训练模型并将ModelCheckpoint回调传递给该模型，得到检查点文件集合，用于分享权重
# - 检查点回调选项：该回调提供了多个选项，用于为生成的检查点提供独一无二的名称，以及调整检查点创建频率。
#
# ### 手动保存权重
# 使用 Model.save_weights 方法即可手动保存权重
#
# ### 保存整个模型
# 整个模型可以保存到一个文件中，其中包含权重值、模型配置（架构）、优化器配置。
# 可以为模型设置检查点，并稍后从完全相同的状态继续训练，而无需访问原始代码。
# Keras通过检查架构来保存模型，使用HDF5标准提供基本的保存格式。
# 特别注意：
# - 目前无法保存TensorFlow优化器（来自tf.train）。
# - 使用此类优化器时，需要在加载模型后对其进行重新编译，使优化器的状态变松散。
