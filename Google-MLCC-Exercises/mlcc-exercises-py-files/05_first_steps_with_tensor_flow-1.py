# coding=utf-8
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ### 设置
tf.logging.set_verbosity(tf.logging.ERROR)  # 日志
pd.options.display.max_rows = 10  # 数据显示
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("Zcalifornia_housing_train.csv", sep=",")  # 加载数据集
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))  # 对数据进行随机化处理
california_housing_dataframe["median_house_value"] /= 1000.0  # 将median_house_value调整为以千为单位
print("california_housing_dataframe: ", california_housing_dataframe)

# ### 检查数据
print("california_housing_dataframe description: ", california_housing_dataframe.describe())  # 各列的统计摘要

# ### 构建第一个模型
# 第1步：定义特征并配置特征列
my_feature = california_housing_dataframe[["total_rooms"]]  # 从california_housing_dataframe中提取total_rooms数据
feature_columns = [tf.feature_column.numeric_column("total_rooms")]  # 使用numeric_column定义特征列，将其数据指定为数值
# 第2步：定义目标
targets = california_housing_dataframe["median_house_value"]
# 第3步：配置LinearRegressor
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)  # 使用梯度下降法训练模型
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  # 应用梯度裁剪到优化器
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer)  # 配置linear_regressor


# 第4步：定义输入函数


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):  # 定义输入函数
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    features = {key: np.array(value) for key, value in dict(features).items()}  # 将Pandas特征数据转换成NumPy数组字典
    ds = Dataset.from_tensor_slices((features, targets))  # 数据构建Dataset对象
    ds = ds.batch(batch_size).repeat(num_epochs)  # 将数据拆分成多批数据，以按照指定周期数进行重复

    if shuffle:  # 如果shuffle设置为True，则会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型
        ds = ds.shuffle(buffer_size=10000)  # buffer_size参数指定shuffle从中随机抽样的数据集的大小

    features, labels = ds.make_one_shot_iterator().get_next()  # 为该数据集构建一个迭代器，并向LinearRegressor返回下一批数据
    return features, labels


# 第5步：训练模型
_ = linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets),
    steps=100)  # 在 linear_regressor 上调用 train() 来训练模型

# 第6步：评估模型
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])
mean_squared_error = metrics.mean_squared_error(predictions, targets)  # 均方误差 (MSE)
root_mean_squared_error = math.sqrt(mean_squared_error)  # 均方根误差 (RMSE)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value  # 比较RMSE与目标最大值和最小值的差值
print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

# 根据总体摘要统计信息，预测和目标的符合情况
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print("calibration_data.describe:\n", calibration_data.describe())

# 可视化：根据模型的偏差项和特征权重绘制散点图
sample = california_housing_dataframe.sample(n=300)  # 获得均匀分布的随机数据样本
# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()  # 获得最小值
x_1 = sample["total_rooms"].max()  # 获得最大值
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]  # 特征权重
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')  # 偏差项
# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias  # 在total_rooms最小值下，获得预测值
y_1 = weight * x_1 + bias  # 在total_rooms最大值下，获得预测值
plt.plot([x_0, x_1], [y_0, y_1], c='r')  # 以红色绘制(x_0, y_0)到(x_1, y_1)的回归线
plt.ylabel("median_house_value")  # 标记图形y轴
plt.xlabel("total_rooms")  # 标记图形x轴
plt.scatter(sample["total_rooms"], sample["median_house_value"])  # 从样本数据绘制散点图
plt.show()  # 显示图

# ### 设置
# 加载必要的库；
# 进行必要的设置，例如日志级别、数据显示方式等；
# 加载数据集并做必要的处理；
#   -  对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）
#   -  调整数据的单位，便于模型能够以常用范围内的学习速率较为轻松地学习这些数据
#
# ### 检查数据
# 建议在使用之前，先对数据做初步了解；
# 例如，关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数等；
#
# ### 构建第一个模型
# 尝试预测 median_house_value，它将是标签（有时也称为目标），使用 total_rooms 作为输入特征；
# 使用 TensorFlow Estimator API 提供的 LinearRegressor 接口来训练模型；
# 此 API 负责处理大量低级别模型搭建工作，并会提供执行模型训练、评估和推理的便利方法；
#
# 第1步：定义特征并配置特征列
# 在 TensorFlow 中，使用一种称为“特征列”的结构来表示特征的数据类型；
# 特征列仅存储对特征数据的描述，不包含特征数据本身；
#
# 第2步：定义目标
# 定义目标，也就是 median_house_value
#
# 第3步：配置 LinearRegressor
# 使用 GradientDescentOptimizer（小批量随机梯度下降法 (SGD)）训练该模型，learning_rate参数控制梯度步长的大小;
# 通过 clip_gradients_by_norm 将梯度裁剪应用到优化器，梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败；
#
# 第4步：定义输入函数
# 定义一个输入函数，告诉TensorFlow如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据；
#
# 第5步：训练模型
# 在linear_regressor上调用train()来训练模型；
#
# 第6步：评估模型
# 基于该训练数据做一次预测，观察模型在训练期间与这些数据的拟合情况；
# 注意：训练误差可以衡量模型与训练数据的拟合情况，但并不能衡量模型泛化到新数据的效果；
# - 均方误差 (MSE)与均方根误差 (RMSE)：由于MSE很难解读，因此经常查看的是RMSE，可以在与原目标相同的规模下解读；
# - 根据总体摘要统计信息，预测和目标的符合情况；
# - 将结果可视化
