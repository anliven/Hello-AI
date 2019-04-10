# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version: {}  - tf.keras version: {}".format(tf.VERSION, tf.keras.__version__))  # 查看版本

# ### 数据部分
# 获取数据（Get the data）
ds_path = str(pathlib.Path.cwd()) + "\\datasets\\auto-mpg\\"
ds_file = keras.utils.get_file(fname=ds_path + "auto-mpg.data", origin="file:///" + ds_path)  # 获得文件路径
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(ds_file,  #
                          names=column_names,  #
                          na_values="?",  #
                          comment='\t',  #
                          sep=" ",  #
                          skipinitialspace=True)  # 通过pandas导入数据
dataset = raw_dataset.copy()
print(dataset.tail())  # 显示尾部数据

# 清洗数据（Clean the data）
print(dataset.isna().sum())  # The dataset contains a few unknown values
dataset = dataset.dropna()  # To keep this initial tutorial simple drop those rows.
origin = dataset.pop('Origin')  # The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()
print("data set tail: {}".format(dataset.tail()))  # 显示尾部数据

# 划分训练集和测试集（Split the data into train and test）
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)  # 测试作为模型的最终评估

# 检查数据（Inspect the data）
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.figure(num=1)
plt.savefig("./outputs/sample-3-figure-1.png", dpi=200, format='png')
plt.show()
plt.close()
train_stats = train_dataset.describe()  # 总体统计数据
train_stats.pop("MPG")
train_stats = train_stats.transpose()  #
print("train stats: {}".format(train_stats))

# 分离标签（Split features from labels）
train_labels = train_dataset.pop('MPG')  # 将要预测的值
test_labels = test_dataset.pop('MPG')


# 规范化的数据（Normalize the data）
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# ### 模型部分
# 构建模型（Build the model）
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# 检查模型（Inspect the model）
mod = build_model()
mod.summary()  # 打印出关于模型的简单描述
example_batch = normed_train_data[:10]
example_result = mod.predict(example_batch)
print("Example result: {}\n".format(example_result))


# 训练模型（Train the model）
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000  # 训练次数

history = mod.fit(normed_train_data,
                  train_labels,
                  epochs=EPOCHS,
                  validation_split=0.2,
                  verbose=0,
                  callbacks=[PrintDot()])  # 模型参数


def plot_history(h, n=1):
    hist = pd.DataFrame(h.history)
    hist['epoch'] = h.epoch
    print("\nHistory tail: {}".format(hist.tail()))

    plt.figure(num=n, figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.ylim([0, 5])

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.ylim([0, 20])

    filename = "./outputs/sample-3-figure-" + str(n) + ".png"
    plt.savefig(filename, dpi=200, format='png')
    plt.show()
    plt.close()


plot_history(history, 2)  # 可视化

# 调试
model2 = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=10)  # 指定提前停止训练的callbacks
history2 = model2.fit(normed_train_data,
                      train_labels,
                      epochs=EPOCHS,
                      validation_split=0.2,
                      verbose=0,
                      callbacks=[early_stop, PrintDot()])  # 当没有改进时自动停止训练
plot_history(history2, 3)
loss, mae, mse = model2.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))  # 在验证集上的MAE值

# 做出预测（Make predictions）
test_predictions = model2.predict(normed_test_data).flatten()
plt.figure(num=4, figsize=(6, 8))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.savefig("./outputs/sample-3-figure-4.png", dpi=200, format='png')
plt.show()
plt.close()

# ### Keras与tf.keras
# Keras是一个用于构建和训练深度学习模型的高级API
# TensorFlow中的tf.keras是Keras API规范的TensorFlow实现，可以运行任何与Keras兼容的代码，保留了一些细微的差别
# 最新版TensorFlow中的tf.keras版本可能与PyPI中的最新Keras版本不同
#
# ### Auto MPG Data Set （汽车MPG数据集）
# - mpg（miles per gallon, 每加仑行驶的英里数）
# - https://archive.ics.uci.edu/ml/datasets/Auto+MPG
# - https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/
# Attribute Information:
# 1. mpg: continuous
# 2. cylinders: multi-valued discrete
# 3. displacement: continuous
# 4. horsepower: continuous
# 5. weight: continuous
# 6. acceleration: continuous
# 7. model year: multi-valued discrete
# 8. origin: multi-valued discrete
# 9. car name: string (unique for each instance)
#
# ### EarlyStopping
# 使用一个EarlyStopping回调来测试每一个迭代的训练条件，如果某个迭代过后没有显示改进，自动停止训练
# - https://keras.io/zh/callbacks/
# - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
