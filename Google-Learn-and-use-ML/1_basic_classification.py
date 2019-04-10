# coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version:", tf.__version__)  # TensorFlow版本

# ### 加载数据集
# 网络畅通的情况下，可以从 TensorFlow 直接访问 Fashion MNIST，只需导入和加载数据即可
# 或者手工下载文件，并存放在“~/.keras/datasets”下的fashion-mnist目录
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 训练集：train_images 和 train_labels 数组，用于学习的数据
# 测试集：test_images 和 test_labels 数组，用于测试模型
# 图像images为28x28的NumPy数组，像素值介于0到255之间
# 标签labels是整数数组，介于0到9之间，对应于图像代表的服饰所属的类别，每张图像都映射到一个标签

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # 类别名称

# ### 探索数据：了解数据格式
print("train_images.shape: {}".format(train_images.shape))  # 训练集中有60000张图像，每张图像都为28x28像素
print("train_labels len: {}".format(len(train_labels)))  # 训练集中有60000个标签
print("train_labels: {}".format(train_labels))  # 每个标签都是一个介于 0 到 9 之间的整数
print("test_images.shape: {}".format(test_images.shape))  # 测试集中有10000张图像，每张图像都为28x28像素
print("test_labels len: {}".format(len(test_labels)))  # 测试集中有10000个标签
print("test_labels: {}".format(test_labels))

# ### 预处理数据
# 必须先对数据进行预处理，然后再训练网络
plt.figure(num=1)  # 创建图形窗口，参数num是图像编号
plt.imshow(train_images[0])  # 绘制图片
plt.colorbar()  # 渐变色度条
plt.grid(False)  # 显示网格
plt.savefig("./outputs/sample-1-figure-1.png", dpi=200, format='png')  # 保存文件，必须在plt.show()前使用，否则将是空白内容
plt.show()  # 显示
plt.close()  # 关闭figure实例，如果要创建多个figure实例，必须显示调用close方法来释放不再使用的figure实例

# 值缩小为0到1之间的浮点数
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示训练集中的前25张图像，并在每张图像下显示类别名称
plt.figure(num=2, figsize=(10, 10))  # 参数figsize指定宽和高，单位为英寸
for i in range(25):  # 前25张图像
    plt.subplot(5, 5, i + 1)
    plt.xticks([])  # x坐标轴刻度
    plt.yticks([])  # y坐标轴刻度
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])  # x坐标轴名称
plt.savefig("./outputs/sample-1-figure-2.png", dpi=200, format='png')
plt.show()
plt.close()

# ### 构建模型
# 构建神经网络需要先配置模型的层，然后再编译模型
# 设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 将图像格式从二维数组（28x28像素）转换成一维数组（784 像素）
    keras.layers.Dense(128, activation=tf.nn.relu),  # 全连接神经层，具有128个节点（或神经元）
    keras.layers.Dense(10, activation=tf.nn.softmax)])  # 全连接神经层，具有10个节点的softmax层
# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),  # 优化器：根据模型看到的数据及其损失函数更新模型的方式
              loss='sparse_categorical_crossentropy',  # 损失函数：衡量模型在训练期间的准确率。
              metrics=['accuracy'])  # 指标：用于监控训练和测试步骤；这里使用准确率（图像被正确分类的比例）

# ### 训练模型
# 将训练数据馈送到模型中，模型学习将图像与标签相关联
model.fit(train_images,  # 训练数据
          train_labels,  # 训练数据
          epochs=5,  # 训练周期（训练模型迭代轮次）
          verbose=2  # 日志显示模式：0为安静模式, 1为进度条（默认）, 2为每轮一行
          )  # 调用model.fit 方法开始训练，使模型与训练数据“拟合

# ### 评估准确率
# 比较模型在测试数据集上的表现
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss: {} - Test accuracy: {}'.format(test_loss, test_acc))

# ### 做出预测
predictions = model.predict(test_images)
print("The first prediction: {}".format(predictions[0]))  # 查看第一个预测结果(包含10个数字的数组，分别对应10种服饰的“置信度”
label_number = np.argmax(predictions[0])  # 置信度值最大的标签
print("label: {} - class name: {}".format(label_number, class_names[label_number]))
print("Result true or false: {}".format(test_labels[0] == label_number))  # 对比测试标签，查看该预测是否正确


# 可视化：将该预测绘制成图来查看全部10个通道
def plot_image(m, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[m], true_label[m], img[m]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'  # 正确的预测标签为蓝色
    else:
        color = 'red'  # 错误的预测标签为红色
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(n, predictions_array, true_label):
    predictions_array, true_label = predictions_array[n], true_label[n]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 查看第0张图像、预测和预测数组
i = 0
plt.figure(num=3, figsize=(8, 5))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.xticks(range(10), class_names, rotation=45)  # x坐标轴刻度，参数rotation表示label旋转显示角度
plt.savefig("./outputs/sample-1-figure-3.png", dpi=200, format='png')
plt.show()
plt.close()

# 查看第12张图像、预测和预测数组
i = 12
plt.figure(num=4, figsize=(8, 5))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.xticks(range(10), class_names, rotation=45)  # range(10)作为x轴的刻度，class_names作为对应的标签
plt.savefig("./outputs/sample-1-figure-4.png", dpi=200, format='png')
plt.show()
plt.close()

# 绘制图像：正确的预测标签为蓝色，错误的预测标签为红色，数字表示预测标签的百分比（总计为 100）
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(num=5, figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
plt.savefig("./outputs/sample-1-figure-5.png", dpi=200, format='png')
plt.show()
plt.close()

# 使用经过训练的模型对单个图像进行预测
image = test_images[0]  # 从测试数据集获得一个图像
print("img shape: {}".format(image.shape))  # 图像的shape信息
image = (np.expand_dims(image, 0))  # 添加到列表中
print("img shape: {}".format(image.shape))
predictions_single = model.predict(image)  # model.predict返回一组列表，每个列表对应批次数据中的每张图像
print("prediction_single: {}".format(predictions_single))  # 查看预测，预测结果是一个具有10个数字的数组，分别对应10种不同服饰的“置信度”

plt.figure(num=6)
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.savefig("./outputs/sample-1-figure-6.png", dpi=200, format='png')
plt.show()
plt.close()

prediction_result = np.argmax(predictions_single[0])  # 获取批次数据中相应图像的预测结果（置信度值最大的标签）
print("prediction_result: {}".format(prediction_result))

# ### 基本分类
# 官网示例：https://www.tensorflow.org/tutorials/keras/basic_classification
# 主要步骤：
#   1.加载Fashion MNIST数据集
#   2.探索数据：了解数据集格式
#   3.预处理数据
#   4.构建模型：设置层、编译模型
#   5.训练模型
#   6.评估准确率
#   7.做出预测：可视化
#
# ### tf.keras
# - 一种用于在TensorFlow中构建和训练模型的高阶API：
# - https://www.tensorflow.org/api_docs/python/tf/keras/
#
# ### Fashion MNIST数据集
# - 经典 MNIST 数据集（常用作计算机视觉机器学习程序的“Hello, World”入门数据集）的简易替换
# - 包含训练数据60000个，测试数据10000个，每个图片是28x28像素的灰度图像，涵盖10个类别
# - https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles
# - TensorFlow：https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist
# - GitHub：https://github.com/zalandoresearch/fashion-mnist
#
# ### 过拟合
# 如果机器学习模型在新数据上的表现不如在训练数据上的表现，就表示出现过拟合
