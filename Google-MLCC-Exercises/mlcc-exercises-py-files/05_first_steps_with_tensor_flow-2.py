# coding=utf-8
import math
from IPython import display
from matplotlib import cm
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
pd.options.display.max_rows = 10  # 显示的最大行数
pd.options.display.float_format = '{:.1f}'.format
pd.set_option('display.max_columns', None)  # 显示的最大列数， None表示显示所有列
pd.set_option('display.width', 200)  # 显示宽度（以字符为单位）
pd.set_option('max_colwidth', 100)  # 列长度，默认为50
pd.set_option('expand_frame_repr', False)  # 是否换行显示，False表示不允许， True表示允许

california_housing_dataframe = pd.read_csv("Zcalifornia_housing_train.csv", sep=",")  # 加载数据集
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))  # 对数据进行随机化处理
california_housing_dataframe["median_house_value"] /= 1000.0  # 将median_house_value调整为以千为单位
print("california_housing_dataframe: ", california_housing_dataframe)

# ### 检查数据
print("california_housing_dataframe description: ", california_housing_dataframe.describe())  # 各列的统计摘要


# ### 定义输入函数


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


# ### 调整模型超参数（代码合并为一个函数，便于调试）


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """
    # 定义观察时间段
    periods = 10  #
    steps_per_period = steps / periods  # 在10个等分的时间段内使用此函数，以便观察模型在每个时间段的改善情况

    # 定义特征并配置特征列
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]  # 从california_housing_dataframe中提取total_rooms数据
    feature_columns = [tf.feature_column.numeric_column(my_feature)]  # 使用numeric_column定义特征列，将其数据指定为数值

    # 定义目标
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # 配置LinearRegressor
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer)

    # 创建输入函数
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # 可视化：模型在每个时间段的状态
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])  # 从样本数据绘制散点图
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # 训练与评估模型：在时间段循环中训练，并定期评估损失
    print("Training model...\nRMSE (on training data):")
    root_mean_squared_errors = []  # 记录损失
    for period in range(0, periods):  # 时间段循环
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period)  # 在 linear_regressor 上调用 train() 来训练模型，从之前的状态开始
        # 评估
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))  # 计算RMSE
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))  # 打印当前时间段的损失：均方根误差 (RMSE)
        root_mean_squared_errors.append(root_mean_squared_error)  # 加入到损失列表

        # 可视化：根据模型的偏差项和特征权重绘制回归线
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]  # 特征权重
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')  # 偏差项
        y_extents = np.array([0, sample[my_label].max()])  # 获得最大值
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])  # 绘制x_extents到y_extents的回归线
    print("Model training finished.")

    # 可视化：损失指标与时间段
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')  # 图形y轴
    plt.xlabel('Periods')  # 图形x轴
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    plt.show()

    # 根据总体摘要统计信息，输出预测和目标的符合情况
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)  # 最终的RMSE值


# ### 任务1：使RMSE不超过180
train_model(
    learning_rate=0.0001,  # 学习速率
    steps=150,  # 步数
    batch_size=2  # 批量
)

# ### 任务2：尝试其他特征
train_model(
    learning_rate=0.0001,
    steps=200,
    batch_size=10,
    input_feature="population"  # 特征
)

# ### 调整模型超参数
# 调整模型超参数，以降低损失和更符合目标分布；
# 使用不同的参数，以了解相应效果；
# 本例是在10个等分的时间段内使用此函数，以便观察模型在每个时间段的改善情况；
# 对于每个时间段都会计算训练损失并绘制相应图表，帮助判断模型收敛的时间，或者模型是否需要更多迭代；
# 绘制模型随着时间的推移学习的特征权重和偏差项值的曲线图，可以查看模型的收敛效果；
#
# ### 注意事项
# 不同超参数的效果取决于数据，不存在必须严格遵循的规则，因此必须始终进行实现和验证；
# 可借鉴的经验法则：
# - 训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。
# - 如果训练尚未收敛，尝试运行更长的时间。
# - 如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。但有时如果学习速率过高，训练误差的减小速度反而会变慢。
# - 如果训练误差变化很大，尝试降低学习速率。较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。
# - 批量大小过小也会导致不稳定情况。不妨先尝试100或1000等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。
