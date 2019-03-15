# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # NumPy是一种用于进行科学计算的常用工具包

print("version: ", pd.__version__)

# ### 显示设置
# 默认情况下，如果DataFrame的行列数量太多，print将只显示部分内容
pd.set_option('display.max_rows', None)  # 显示的最大行数，None表示显示所有行
pd.set_option('display.max_columns', None)  # 显示的最大列数， None表示显示所有列
pd.set_option('display.width', 200)  # 显示宽度（以字符为单位）
pd.set_option('max_colwidth', 100)  # 列长度，默认为50
pd.set_option('expand_frame_repr', False)  # 是否换行显示，False表示不允许， True表示允许

# ### 基本概念
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])  # 构建Series对象
population = pd.Series([852469, 1015785, 485199])  # 构建Series对象
maps = pd.DataFrame({'City name': city_names, 'Population': population})  # 创建DataFrame对象
print("DataFrame: ", "\n", maps)  # 如果Series在长度上不一致，系统会用特殊的NA/NaN值填充缺失的值

california_housing_dataframe = pd.read_csv("Zcalifornia_housing_train.csv", sep=",")  # 将整个文件加载到DataFrame
info = california_housing_dataframe.describe()  # 使用DataFrame.describe来显示关于DataFrame的统计信息
print(info)
info_head = california_housing_dataframe.head()  # 显示DataFrame的前几个记录
print(info_head)

hist = california_housing_dataframe.hist('housing_median_age')  # 绘制图表：使用DataFrame.hist快速了解一个列中值的分布
print(hist)
plt.show()  # 显示图表

# ### 访问数据
# 以Python的dict/list方式访问DataFrame数据
cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(type(cities['City name']), type(cities[0:2]), type(cities['City name'][1]))
print(cities)
print(cities['City name'])
print(cities[0:2])
print(cities['City name'][1])

# ### 操控数据
print(population / 1000)  # 可以向Series应用Python的基本运算指令
print(np.log(population))  # Series对象可用作大多数NumPy函数的参数
print(population.apply(lambda val: val > 1000000))  # 创建一个population是否超过100万的新Series对象
# 使用Series.apply进行复杂的单列转换，Series.apply将以参数形式接受 lambda 函数，而该函数会应用于每个值
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])  # 向现有DataFrame添加Series
cities['Population density'] = cities['Population'] / cities['Area square miles']  # 添加Series
print(cities)

# 练习
cities['wide_saint'] = (cities['Area square miles'] > 50) \
                       & cities['City name'].apply(lambda name: name.startswith('San'))  # 添加Series
# 注意：布尔值Series是使用“按位”而非传统布尔值“运算符”组合的，因此执行逻辑与时，应使用&，而不是and
print(cities)

# ### 索引
# Series和DataFrame对象定义了index属性，该属性会向每个Series项或DataFrame行赋一个标识符值
# 默认情况下，在构造时，pandas会赋可反映源数据顺序的索引值
# 索引值在创建后是稳定的，不会因为数据重新排序而发生改变
print(city_names.index)
print(cities.index)
print(cities.reindex([2, 0, 1]))  # 调用DataFrame.reindex来手动重新排列各行的顺序
cities.reindex(np.random.permutation(cities.index))  # 将cities.index传递至NumPy的random.permutation函数，随机排列其值的位置
print(cities)

# ### 练习
# reindex方法允许使用未包含在原始DataFrame索引值中的索引值，reindex会为此类“丢失的”索引添加新行，并在所有对应列中填充NaN值
print(cities.reindex([0, 4, 5, 2]))

# ### Pandas
# - HomePage: http://pandas.pydata.org/
# - 针对Python语言的开源数据分析处理工具，可以提供高性能、易用的数据结构；
#
# ### 官网文档
# - Docs: http://pandas.pydata.org/pandas-docs/stable/index.html
# - Function Index：http://pandas.pydata.org/pandas-docs/stable/genindex.html
# - API：http://pandas.pydata.org/pandas-docs/stable/api.html
# - Tutorials：http://pandas.pydata.org/pandas-docs/stable/tutorials.html
# - 10 Minutes to pandas：http://pandas.pydata.org/pandas-docs/stable/10min.html
# - Cookbook：http://pandas.pydata.org/pandas-docs/stable/cookbook.html
#
# ### 主要数据结构
# - DataFrame: 数据框架是用于数据操控的一种常用抽象实现形式，可以理解为一个关系型数据表格，其中包含多个行和已命名的列。
# - Series: 单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。
