# Big-Data-Machine-Learning-Task-2
### Research on the prediction model of carbon dioxide concentration in pigsty based on EEMD-GRU 基于EEMD-GRU的猪舍二氧化碳浓度预测模型研究
大数据及农业应用-课程作业\
大三上


### 一、实验目的
基于EEMD-GRU的猪舍二氧化碳浓度预测模型研究\
本实验在农业农村部饲料工业中心动物试验基地妊娠猪舍进行了数据采集，主要采集猪舍内温度、湿度、风速和二氧化碳浓度数据。\
猪舍地点位于河北省承德市丰宁满族自治县，实验猪舍面积为12×8m2，采用双排限位栏和水泥地面，饲养30头妊娠前期的母猪，采用头对头饲养方式，中间是过道，后面是排水沟。\
猪舍环境传感器购于山东省青岛大牧人机械股份有限公司，二氧化碳传感器测量范围为0-5000ppm，湿度传感器采用HTV597型号，环境控制器采用BH8118型号，可监测显示并调控猪舍内温度、湿度和风速值等。\
基于以上背景对猪舍二氧化碳浓度进行分析预测。

### 二、实验原理
（1）数据收集\
![image](https://github.com/user-attachments/assets/cef3b6ef-3b43-4c33-84c1-326a8ef26c8e)\
环境监测系统\
（2）数据预处理：\
• 小时均值处理：\
整理传感器每10分钟采集到的数据，将1小时内测量到的6次数据加和平均处理，便于后续的数据处理和模型建立。\
小时均值处理公式：\
![image](https://github.com/user-attachments/assets/3b6200bd-8d83-4fab-bbfc-94ea1ef25816)\
式中：xh是小时均值处理后数据，xi是每10分钟各采样点数据。\
• 数据归一化：\
为提高算法收敛速度和精度，使模型建立、学习、训练和预测的效果更好，需要对数据进行标准化处理。\
本实验采用数据归一化方法中的最大最小值归一化法，即线性函数归一化法。其原理是：通过使用数据集中数据的最大值和最小值进行标准化处理，使得处理后的数据集中在大于0小于1的区间范围内，具体公式为：
![image](https://github.com/user-attachments/assets/852615b2-253f-4313-9b7d-957c8e45bf81)\
式中：X∗为归一化处理后数据，X是采集的环境参数，Xmax、Xmin是环境参数中最大值与最小值。

### 三、实验步骤
（1）时间序列与监督学习：\
在可以使用机器学习之前，时间序列预测问题必须重新构建成监督学习问题，从一个单纯的序列变成一对序列输入和输出。\
定义一个名为series_to_supervised( )的新Python函数，它采用单变量或多变量时间序列，并将其作为监督学习数据集。\
该函数有四个参数：\
• data：序列，列表或二维数组。\
• n_in：用于输入数据步数(x)。值可能介于[1,len(data)]，可选参数。\
• n_out：作为输出数据步数(y)。值可能介于[1,len(data)]，可选参数。\
• dropnan：用于滤除缺失数据。可选参数。默认为True。\
代码实现：\
• 首先使用MinMaxScaler( )函数对数据进行归一化处理。\
• 然后通过series_to_supervised( )函数将数据转换为有监督的数据。\
• 最后利用drop( )函数删除不预测的列。\
（2）数据集制作：\
将前两小时的环境数据以及二氧化碳数据进行整合作为一个样本，所以模型的输入特征数据分别是（共计8维数据特征）：\
• 两小时前的二氧化碳浓度\
• 两小时前的温湿度\
• 两小时前的风速\
• 一小时前的二氧化碳浓度\
• 一小时前的温湿度\
• 一小时前的风速\
（3）数据集划分：\
• 将数据集70%作为训练集，30%作为测试集。\
• 通过reshape( )函数将训练集与测试集转化为3维，三个参数分别为：数据集行数(shape[0])、输入序列步数( n_in )、特征数(feature)\
（4）对EEMD进行分解。\
![image](https://github.com/user-attachments/assets/2696a5dc-b50b-40c4-8054-53d8d879c1ea)\
EEMD分解结果\
（5）GRU模型构建：GRU模型参数设置如下，\
• 隐藏层数为1，\
• 神经元个数为35，\
• 输出层维度为1，\
• Epoch为200，\
• Batch_size为128，\
• 损失函数为mae，\
• 优化器可选adam优化器。\
（6）GRU模型调参：根据需求对神经元个数及网络层数进行选择。\
此外，可对Batch_size、学习率等参数进行优化。\
为防止过拟合，可采用Dropout方法，随机选择神经层中的一些单元并将其临时隐藏。\
（7）评价与绘图：将RMSE（root mean squared error）与MAE（mean absolute error）作为评价指标，最后绘制预测值与真实值曲线图。

### 四、实验结果
![image](https://github.com/user-attachments/assets/a0df22c8-5fd8-46f3-a728-20385bf4be67)\
EEMD-GRU.py运行出的CO2预测值和测量值对比图\
![image](https://github.com/user-attachments/assets/6bb87053-d4d0-44ce-96b6-33bd6cef18de)\
WinBP.py运行出的Model loss训练集和测试集结果\
![image](https://github.com/user-attachments/assets/aa26a71a-36da-4dec-8c89-dcc226c3040e)\
WinBP.py运行出的BP模型的CO2预测值和测量值对比图\
![image](https://github.com/user-attachments/assets/bba64dd9-b80b-401f-9c92-8af0736e2b3a)\
GRU.py运行后的训练集和测试集结果\
![image](https://github.com/user-attachments/assets/593d833c-8aa2-4221-b3da-81fc9c726959)\
在epochs=200, batch_size=128参数下GRU对CO2预测值和真实值对比图\
![image](https://github.com/user-attachments/assets/c03f1996-db8f-4e4d-8603-2c2457b02502)\
在epochs=400, batch_size=256参数下GRU对CO2预测值和真实值对比图\
![image](https://github.com/user-attachments/assets/720003d0-ac41-4236-9f24-dcdd316cc64e)\
在epochs=100, batch_size=64参数下GRU对CO2预测值和真实值对比图

### 五、实验总结
本研究通过构建基于EEMD-GRU的猪舍二氧化碳浓度预测模型，实现了对猪舍内二氧化碳浓度的准确预测。实验结果表明，该模型具有较高的预测精度和稳定性，能够为猪舍环境管理提供科学依据。\
通过EEMD分解，实验成功提取了原始数据中的潜在特征，为后续的GRU模型构建提供了有力的支持。同时，GRU模型凭借其强大的时间序列处理能力，有效地捕捉了猪舍内二氧化碳浓度的变化趋势。在epochs=200, batch_size=128参数下GRU预测值和真实值最为接近，在epochs=400, batch_size=256参数下和在epochs=100, batch_size=64参数下GRU预测值和真实值相对较差。说明在epochs=200, batch_size=128参数下GRU更能够准确预测出猪舍内二氧化碳浓度。\
本研究通过模型调参等一系列措施，进一步提高了模型的预测性能。
