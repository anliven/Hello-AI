# 1 - MLCC
通过机器学习，可以有效地解读数据的潜在含义，甚至可以改变思考问题的方式，使用统计信息而非逻辑推理来处理问题。
Google的机器学习速成课程（MLCC，machine-learning crash-course）：https://developers.google.com/machine-learning/crash-course/
支持多语言，共25节课程，包含40多项练习，有对算法实际运用的互动直观展示，可以更容易地学习和实践机器学习概念。
官方预估时间大约15小时(实际花费时间根据个人情况而定，差异较大)。

注意：这里的时间长度指的是教程播放和阅读的时间，而不是你“真正”理解吸收和练习的时间。
实际上，对于“小白”阶段的新手，可能要投入数倍于此的精力，才能完成整个学习过程（观看、阅读、理解、练习、了解相关知识点、等等）。

**本课程将解答如下问题：**
- 机器学习与传统编程有何不同？
- 什么是损失，如何衡量损失？
- 梯度下降法的运作方式是怎样的？
- 如何确定我的模型是否有效？
- 怎样为机器学习提供我的数据？
- 如何构建深度神经网络？

**机器学习术语表**
- https://developers.google.com/machine-learning/glossary/    
- PDF文件：[下载](https://files.cnblogs.com/files/anliven/GoogleMLCC%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%9C%AF%E8%AF%AD%E8%A1%A8.zip)


## 1.1 - 练习环境
MLCC相关练习：https://developers.google.com/machine-learning/crash-course/exercises

### 本地运行
- https://developers.google.com/machine-learning/crash-course/running-exercises-locally
- 下载练习：http://download.mlcc.google.com/mledu-exercises/mlcc-exercises_en.zip
- 安装Anaconda（包含Jupyter notebook），创建环境并运行Jupyter Notebook (.ipynb) 格式的编程练习。
例如：
```
(base) >conda create -n mlcc pip python=3.6
(base) >conda activate mlcc
(mlcc) >pip --proxy="10.144.1.10:8080" install --ignore-installed --upgrade tensorflow matplotlib pandas sklearn scipy seaborn  # pip使用代理
(mlcc) >pip install --ignore-installed --upgrade tensorflow matplotlib pandas sklearn scipy seaborn -i https://mirrors.ustc.edu.cn/pypi/web/simple/  # pip使用国内源
```

### 在线运行
- Colaboratory： https://colab.research.google.com/notebooks/welcome.ipynb
- 免费的Jupyter笔记本环境，直接在浏览器中运行编程练习，不需要进行任何设置就可以使用，并且完全在云端运行。


## 1.2 - 前提条件和准备工作
### 前提条件
```
掌握入门级代数知识。 
    了解变量和系数、线性方程式、函数图和直方图（熟悉对数和导数等更高级的数学概念会有帮助，但不是必需条件）。

熟练掌握编程基础知识，并且具有一些使用 Python 进行编码的经验。
    机器学习速成课程中的编程练习是通过TensorFlow并使用Python进行编码的。
    无需拥有任何 TensorFlow经验，但应该能够熟练阅读和编写包含基础编程结构（例如，函数定义/调用、列表和字典、循环和条件表达式）的Python代码。
```

### 准备工作
```
Pandas 使用入门
    机器学习速成课程中的编程练习使用 Pandas 库来操控数据集。
    如果不熟悉 Pandas，建议先学习Pandas 简介教程，该教程介绍了练习中使用的主要 Pandas 功能。

低阶 TensorFlow 基础知识
   机器学习速成课程中的编程练习使用 TensorFlow 的高阶 tf.estimator API 来配置模型。
   如果有兴趣从头开始构建 TensorFlow 模型，请学习以下教程：
     - TensorFlow Hello World：在低阶 TensorFlow 中编码的“Hello World”。
     - TensorFlow 编程概念：演示了 TensorFlow 应用中的基本组件：张量、指令、图和会话。
     - 创建和操控张量：张量快速入门 - TensorFlow 编程中的核心概念。此外，还回顾了线性代数中的矩阵加法和乘法概念。
```


## 1.3 - 主要概念和工具
### 数学
```
代数
    - 变量、系数和函数
    - 线性方程式
    - 对数和对数方程式
    - S型函数

线性代数
    - 张量和张量等级
    - 矩阵乘法

三角学
    - Tanh（作为激活函数进行讲解，无需提前掌握相关知识）

统计信息
    - 均值、中间值、离群值和标准偏差
    - 能够读懂直方图

微积分（可选，适合高级主题）
    - 导数概念（您不必真正计算导数）
    - 梯度或斜率
    - 偏导数（与梯度紧密相关）
    - 链式法则（带您全面了解用于训练神经网络的反向传播算法）
```

### Python（https://docs.python.org/3/tutorial/）
```
基础 Python
    - 定义和调用函数：使用位置和关键字参数
    - 字典、列表、集合（创建、访问和迭代）
    - for 循环：包含多个迭代器变量的 for 循环（例如 for a, b in [(1,2), (3,4)]）
    - if/else 条件块和条件表达式
    - 字符串格式（例如 '%.2f' % 3.14）
    - 变量、赋值、基本数据类型（int、float、bool、str）
    - pass 语句
中级 Python
    - 列表推导式
    - Lambda 函数
```

### 第三方Python库（无需提前熟悉，在需要时查询相关内容）
```
Matplotlib（适合数据可视化）
    - pyplot 模块
    - cm 模块
    - gridspec 模块

Seaborn（适合热图）
    - heatmap 函数

Pandas（适合数据处理）
    - DataFrame 类

NumPy（适合低阶数学运算）
    - linspace 函数
    - random 函数
    - array 函数
    - arange 函数

Scikit-Learn（适合评估指标）
   - metrics 模块
```

### Bash
- Bash参考手册：https://tiswww.case.edu/php/chet/bash/bashref.html
- Bash快速参考表：https://github.com/LeCoupa/awesome-cheatsheets/blob/master/languages/bash.sh
- 了解Shell（简明教程，提供在线运行环境）：http://www.learnshell.org/



# 2 - 下一步
## 2.1 - Google的机器学习指南
https://developers.google.com/machine-learning/guides/
通过简单的逐步演示介绍如何利用最佳做法解决常见的机器学习问题。
- 机器学习规则 (Rules of Machine Learning)： https://developers.google.com/machine-learning/guides/
- 文本分类：https://developers.google.com/machine-learning/guides/text-classification/


## 2.2 - Machine Learning Practica
Google的机器学习实践：https://developers.google.com/machine-learning/practica/
- 图片分类：https://developers.google.com/machine-learning/practica/image-classification/
- 了解 Google 如何开发用于在 Google 照片中为搜索提供支持的图片分类模型，然后构建您自己的图片分类器。


## 2.3 - 深入了解官网文档 
以TensorFlow为例：
- https://www.tensorflow.org/guide/
- https://www.tensorflow.org/tutorials/
