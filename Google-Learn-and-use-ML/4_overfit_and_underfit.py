# coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("# TensorFlow version: {}  - tf.keras version: {}".format(tf.VERSION, tf.keras.__version__))  # 查看版本
ds_path = str(pathlib.Path.cwd()) + "\\datasets\\imdb\\"  # 数据集路径

# ### 查看numpy格式数据
np_data = np.load(ds_path + "imdb.npz")
print("# np_data keys: ", list(np_data.keys()))  # 查看所有的键
# print("# np_data values: ", list(np_data.values()))  # 查看所有的值
# print("# np_data items: ", list(np_data.items()))  # 查看所有的item

# ### 加载IMDB数据集
NUM_WORDS = 10000

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    path=ds_path + "imdb.npz",
    num_words=NUM_WORDS  # 保留训练数据中出现频次在前10000位的字词
)


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])  # 查看其中的一个多热向量（字词索引按频率排序，因此索引0附近应该有更多的1值）
plt.savefig("./outputs/sample-4-figure-1.png", dpi=200, format='png')
plt.show()
plt.close()

# ### 演示过拟合
# 创建基准模型
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])  # 仅使用Dense层创建一个简单的基准模型
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])  # 编译模型
baseline_model.summary()  # 打印出关于模型的简单描述
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)  # 训练模型

# 创建一个更小的模型（隐藏单元更少）
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)  # 使用相同的数据训练

# 创建一个更大的模型（远超出解决问题所需的容量）
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])
bigger_model.summary()
bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)  # 使用相同的数据训练


# 绘制训练损失和验证损失图表
# 这里实线表示训练损失，虚线表示验证损失(验证损失越低，表示模型越好)
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))
    for name, his in histories:
        val = plt.plot(his.epoch,
                       his.history['val_' + key],
                       linestyle='--',  # 默认颜色的虚线（dashed line with default color）
                       label=name.title() + ' Val')
        plt.plot(his.epoch,
                 his.history[key],
                 color=val[0].get_color(),
                 label=name.title() + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(his.epoch)])


plot_history([('baseline', baseline_history), ('smaller', smaller_history), ('bigger', bigger_history)])
plt.savefig("./outputs/sample-4-figure-2.png", dpi=200, format='png')
plt.show()
plt.close()

# ### 策略
# 添加权重正则化
l2_model = keras.models.Sequential([
    keras.layers.Dense(16,
                       kernel_regularizer=keras.regularizers.l2(0.001),  # 添加L2权重正则化
                       activation=tf.nn.relu,
                       input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16,
                       kernel_regularizer=keras.regularizers.l2(0.001),  # 实际上是将权重正则化项实例作为关键字参数传递给层
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)])
l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])
l2_model_history = l2_model.fit(train_data,
                                train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)  # 使用相同的数据训练
plot_history([('baseline', baseline_history), ('l2', l2_model_history)])
plt.savefig("./outputs/sample-4-figure-3.png", dpi=200, format='png')
plt.show()  # 查看在训练阶段添加L2正则化惩罚的影响，此过拟合抵抗能力强于基准模型
plt.close()

# 添加丢弃层
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),  # 通过丢弃层将丢弃引入网络中，以便事先将其应用于层的输出
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),  # 添加丢弃层，“丢弃率”设置在0.5
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)
plot_history([('baseline', baseline_history), ('dropout', dpt_model_history)])
plt.savefig("./outputs/sample-4-figure-4.png", dpi=200, format='png')
plt.show()
plt.close()

# ### 过拟合与欠拟合（Overfitting and underfitting）
# 官网示例：https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
# 主要步骤：
#   # 演示过拟合
#   - 创建基准模型
#   - 创建一个更小的模型
#   - 创建一个更大的模型
#   - 绘制训练损失和验证损失函数
#   # 策略
#   - 添加权重正则化
#   - 添加丢弃层
#
# ### 过拟合
# 在训练集上可以实现很高的准确率，但无法很好地泛化到测试数据（或之前未见过的数据）。
# 可能导致欠拟合的原因：训练时间过长等。
# 防止过拟合的最常见方法：
#   - 推荐：获取并使用更多训练数据
#   - 最简单：适当缩小模型（降低网络容量）
#   - 添加权重正则化（限制网络的复杂性，也就是限制可存储信息的数量和类型）
#   - 添加丢弃层
# 此外还有两个重要的方法：数据增强和批次归一化。
#
# ### 欠拟合
# 与过拟合相对的就是欠拟合，测试数据仍存在改进空间，意味着模型未学习到训练数据中的相关模式。
# 可能导致欠拟合的原因：模型不够强大、过于正则化、或者根本没有训练足够长的时间等。
#
# ### 模型大小
# 防止过拟合，最简单的方法是缩小模型，即减少模型中可学习参数的数量（由层数和每层的单元数决定）。
# 在深度学习中，模型中可学习参数的数量通常称为模型的“容量”。
# 模型“记忆容量”越大，越能轻松学习训练样本与其目标之间的字典式完美映射（无任何泛化能力的映射），但无法对未见过的数据做出预测。
# 也就是说，网络容量越大，便能够越快对训练数据进行建模（产生较低的训练损失），但越容易过拟合（导致训练损失与验证损失之间的差异很大）。
# 如果模型太小（记忆资源有限），便无法轻松学习映射，难以与训练数据拟合。
# 需要尝试不断地尝试来确定合适的模型大小或架构（由层数或每层的合适大小决定）。
# 最好先使用相对较少的层和参数，然后开始增加层的大小或添加新的层，直到看到返回的验证损失不断减小为止。
#
# ### 奥卡姆剃刀定律
# 如果对于同一现象有两种解释，最可能正确的解释是“最简单”的解释，即做出最少量假设的解释。
# 也适用于神经网络学习的模型：给定一些训练数据和一个网络架构，有多组权重值（多个模型）可以解释数据，而简单模型比复杂模型更不容易过拟合。
# 简单模型”是一种参数值分布的熵较低的模型（或者具有较少参数的模型）。
#
# ### 权重正则化
# 限制网络的复杂性，具体方法是强制要求其权重仅采用较小的值，使权重值的分布更“规则”。
# 通过向网络的损失函数添加与权重较大相关的代价来实现。
# - L1正则化，其中所添加的代价与权重系数的绝对值（即所谓的权重“L1 范数”）成正比。
# - L2正则化，其中所添加的代价与权重系数值的平方（即所谓的权重“L2 范数”）成正比，也称为权重衰减。
#
# ### 丢弃层
# 添加丢弃层可明显改善基准模型，是最有效且最常用的神经网络正则化技术之一。
# 丢弃（应用于某个层）是指在训练期间随机“丢弃”（即设置为 0）该层的多个输出特征。
# “丢弃率”指变为 0 的特征所占的比例，通常设置在 0.2 和 0.5 之间。
# 因为“测试时的活跃单元数大于训练时的活跃单元数”，因此测试时网络不会丢弃任何单元，而是将层的输出值按等同于丢弃率的比例进行缩减。
