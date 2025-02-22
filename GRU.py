from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import initializers
from keras.regularizers import l2
from keras import optimizers
from math import sqrt
import numpy as np
import random
from numpy.random import seed
seed(1)

def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

#数据集设置索引  绘图出每一种输入的变化趋势
#load data
# def parse(x):
#     return datetime.strptime(x, '%Y-%m-%d %H:%M')#%代表不一样的意思可查表
# dataset = read_csv('mypollution3.csv',  parse_dates = True, index_col=0, date_parser=parse)
# dataset.index.name = 'DATE'#规定索引名字是“DATE”
# dataset = dataset[24:]
# # summarize first 5 rows  打印前五行
# print(dataset.head(5))
# # save to file
# # dataset.to_csv('mypollution3_.csv')
# values = dataset.values
# # # specify columns to plot
# groups = [0, 1, 2]
# i = 1
# # plot each column  每列数据绘图
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])#第group列的所有行
#     pyplot.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()
#
#转换为时间序列的数据(用前两个小时预测后一小时)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_in 表示是根据几行来进行预测
    # n_out 表示是根据前面数据来预测接下来几行的
    #所以这里就是根据前一行的来预测后一行（前一分钟预测后一分钟）
    n_vars = 1 if type(data) is list else data.shape[1] #n_vars是变量数量
    print (n_vars)
    df = DataFrame(data)#DataFrame：一个表格型的数据结构，包含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔型等)，DataFrame即有行索引也有列索引，可以被看做是由Series组成的字典。
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):#n_in 倒着取 n_in，n_in-1，n_in-2....1
        #print("n_in",i)
        #print(df.shift(i))
        cols.append(df.shift(i))#shift函数是对数据进行移动的操作 在列表末尾添加新的对象
        print("i", cols)
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        print(i)
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    #print(cols)
    agg = concat(cols, axis=1) #连接数组
    #print('agg',agg)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
dataset = read_csv('zong1_.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 2, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11]], axis=1, inplace=True) #[7,8]
print(reframed.head())


#数据分割
values = reframed.values
v_train_size=len(values)
print('改变后的训练集数量',v_train_size)
train_size = int(len(dataset) * 0.7)  #70%作为训练集
print('训练集数量',train_size)
train = values[:train_size, :] #矩阵前面是行，后面是列 所以是取所有列
test = values[train_size:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]  #最后一列作为输出标签
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
#其中samples是训练序列的数目，timesteps是序列长度，features是每个序列的特征数目。
train_X = train_X.reshape((train_X.shape[0], 2, 4)) #进行维度变换
test_X = test_X.reshape((test_X.shape[0], 2, 4))
print('序列长度',train_X.shape[0])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)  #.shape 查看数据的维数
print(train_X)

#模型建立
model = Sequential()
model.add(GRU(20, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, kernel_regularizer=l2(0.001),
              recurrent_regularizer=l2(0.005)))
model.add(GRU(20, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.005)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(test_X, test_y))
# model.add(GRU(20, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))#定义输出层神经元个数为1个，即输出只有1维
# model.add(Dropout(0.1))
# sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.8, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
# # fit network
#history = model.fit(train_X, train_y, epochs=1000, batch_size=120, validation_data=(test_X, test_y), verbose=2,
#                    shuffle=False)
#plot history  绘制训练过程损失函数
pyplot.plot(history.history['loss'], label='train',color='#6699FF')
pyplot.plot(history.history['val_loss'], label='test',color='#FF6666')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 8))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -3:]), axis=1)  #预测的
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0] #取第一列，即为二氧化碳的浓度
print("预测",inv_yhat)
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -3:]), axis=1) #真实的
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
#print('真实值',inv_y)

#calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
MAE=mae_value(inv_y, inv_yhat)
print('Test MAE: %.3f' % MAE)
#x=np.arange(618,881,1)#绘制横坐标
pyplot.plot(inv_yhat,label='prediction value',color='#FF6666')
pyplot.plot(inv_y,label='detection value',color='#6699FF')
pyplot.title('GRU model of carbon dioxide value in winter') 
pyplot.xlabel('Time(hour)')
pyplot.ylabel('CO2 Concentration(ppm)')
pyplot.legend()
pyplot.show()
