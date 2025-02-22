# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 00:16:01 2021

@author: yeshuqin
"""

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np #起别名
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import operator
from functools import reduce
from tkinter import _flatten
from pandas import Series  # 从什么导入什么
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据
#dataset = pd.read_csv('mypollution4_.csv', header=0, index_col=0)
dataset = pd.read_csv('zong1_.csv', header=0, index_col=0)
dataset_columns = dataset.columns
values = dataset.values
values = values.astype('float32')
train_size = int(len(dataset) * 0.7)  # 70%作为训练集，能改吗？
train = values[:train_size, :]  # 矩阵前面是行，后面是列 所以是取所有列
test = values[train_size:, :]
# split into input and outputs

train_X, train_y = np.hstack((train[:-2, :],train[1:-1, :])), train[2:, 0]  # 最后一列作为输出标签
test_X, test_y = np.hstack((test[:-2, :],test[1:-1, :])), test[2:,0]
print('训练集输入',train_X)
print('测试集输入',train_y)

# 转成DataFrame格式方便数据处理
train_X_pd = pd.DataFrame(train_X)
train_y_pd = pd.DataFrame(train_y)
test_X_pd = pd.DataFrame(test_X)
test_y_pd = pd.DataFrame(test_y)
#print(x_train_pd.head(5))
#print('-------------------')
#print(y_train_pd.head(5))


# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train_X_pd)
train_X = min_max_scaler.transform(train_X_pd)

min_max_scaler.fit(train_y_pd)
train_y = min_max_scaler.transform(train_y_pd)

# 验证集归一化
min_max_scaler.fit(test_X_pd)
test_X = min_max_scaler.transform(test_X_pd)

min_max_scaler.fit(test_y_pd)
test_y = min_max_scaler.transform(test_y_pd)
print(train_X_pd.shape[1])

model = Sequential()  # 初始化，很重要！
model.add(Dense(units = 10,   # 输出大小
                activation='relu',  # 激励函数
                input_shape=(train_X_pd.shape[1],)  # 输入大小, 也就是列的大小
               )
         )
model.add(Dropout(0.2))  # 丢弃神经元链接概率
model.add(Dense(units = 15,
  #               kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
 #               activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                activation='relu' # 激励函数
 #               bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
               )
         )
model.add(Dense(units = 1,
                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数
               )
         )

model.compile(loss='mae',  # 损失均方误差
              optimizer='adam',  # 优化器
             )

history = model.fit(train_X, train_y,
          epochs=200,  # 迭代次数
          batch_size=128,  # 每次用来梯度下降的批处理数据大小
          verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (test_X, test_y)  # 验证集
        )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 预测
y_new = model.predict(test_X)
# 反归一化
min_max_scaler.fit(test_y_pd)
y_new = min_max_scaler.inverse_transform(y_new)
test_y=min_max_scaler.inverse_transform(test_y)
print(len(y_new))
print(len(test_y))
y_new = y_new[:, 0]
y_new = np.array(y_new)
#x=np.arange(0,800,1)
plt.plot(y_new,label='prediction value',color='#FF6666')
plt.plot(test_y,label='detection value',color='#6699FF')
plt.xlabel('Time(hour)')
plt.ylabel('CO2 Concentration(ppm)')
plt.title('BP model of carbon dioxide value in winter')
plt.legend()
plt.savefig('./Test result of BP.jpg')#保存结果图到本地，必须写在plt.show()前面
plt.show()
rmse = sqrt(mean_squared_error(y_new,test_y ))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(y_new,test_y )
print('Test MAE: %.3f' % mae)
