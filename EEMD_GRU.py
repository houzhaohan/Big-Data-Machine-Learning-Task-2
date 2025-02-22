import pandas as pd
import numpy as np
from PyEMD import EMD, EEMD, Visualisation
from pyhht.visualization import plot_imfs
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.regularizers import l2
from keras import optimizers
import numpy as np
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from numpy.random import seed #设置随机数生成器，保证每次的结果稳定 参考网址https://www.leiphone.com/category/yanxishe/zt4Dm491Ol58C8Mc.html
seed(1)
# #转换为时间序列的数据(用前两个小时预测后一小时)
# #load data
# def parse(x):
#     return datetime.strptime(x, '%Y/%m/%d %H:%M')#%代表不一样的意思可查表
# dataset = pd.read_csv('zong1.csv',  parse_dates = True, index_col=0, date_parser=parse,encoding='gbk')
# dataset.index.name = 'DATE'#规定索引名字是“DATE”
# dataset = dataset[24:]
# # summarize first 5 rows  打印前五行
# print(dataset.head(5))
# # save to file
# dataset.to_csv('zong1_.csv')
# values = dataset.values
# # # specify columns to plot
# groups = [0, 1, 2]
# i = 1
# # plot each column  每列数据绘图
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])#第group列的所有行
#     plt.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
# plt.show()
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_in 表示是根据几行来进行预测
    # n_out 表示是根据前面数据来预测接下来几行的
    #所以这里就是根据前一行的来预测后一行（前一分钟预测后一分钟）
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
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
#计算数组各元素的绝对值
if __name__ == '__main__':
  dataset = pd.read_csv('zong1_.csv', header=0, index_col=0)
  dataset_columns = dataset.columns
  values = dataset.values
  values = values.astype('float32')
 
# 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
# scaleds = []

  '''
  进行emd分解
  '''
  pollution = values[:, 0]
  N = 2611
  tMin, tMax = 0, 2611
  T = np.linspace(tMin, tMax, N)
  max_imf = -1
  eemd = EEMD()
  eemd.trials = 50
  eemd.noise_seed(12345)
  imfs = eemd.eemd(pollution, T, max_imf)
  imfNo = imfs.shape[0]
  plt.ioff()
  # plt.subplot(6, 2, 1)
  # plt.plot(T, pollution, 'k')
  # #plt.plot(T, pollution, 'r')
  # plt.xlim((tMin, tMax))
  # plt.title("Original signal")

  # for num in range(imfNo):
  #     plt.subplot(6, 2, num + 2)
  #     plt.plot(T, imfs[num], 'c')
  #     plt.xlim((tMin, tMax))
  #     plt.title("Imf " + str(num + 1))
  # plt.subplot(6, 2, 11)
  # plt.plot(T, imfs[8], 'g')
  # plt.xlim((tMin, tMax))
  # plt.title("Res")
  # #plt.savefig('./EEMD.jpg')
  # plt.show()
  # plot_imfs(pollution, np.array(imfs))
  ax = plt.subplot(11, 1, 1)
  plt.subplots_adjust(left=4, bottom=4, right=6, top=6,wspace=0, hspace=1)
  plt.plot(T, pollution, '#99FF33')
  #plt.plot(T, pollution, 'r')
  plt.xlim((tMin, tMax))
  plt.title("Original signal",fontsize='xx-large',fontweight='heavy',color='#FF4500')


  for num in range(imfNo):
      plt.subplot(11, 1, num + 2)
      plt.plot(T, imfs[num], '#6699FF')
      plt.xlim((tMin, tMax))
      plt.title("Imf " + str(num + 1),fontsize='xx-large',fontweight='heavy',color='#FF4500')
  plt.subplot(11, 1, 11)
  plt.plot(T, imfs[9], '#6699FF')
  plt.xlim((tMin, tMax))
  plt.title("Res",fontsize='xx-large',fontweight='heavy',color='#FF4500')
  #plt.savefig('./EEMD.jpg')
  plt.show()
  #plot_imfs(pollution, np.array(imfs))
  imfsValues = []
  for imf in imfs:
    values[:, 0] = imf
    imfsValues.append(values.copy())
  inv_yHats = []
  inv_ys  = []
  for imf in imfsValues:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(imf)
    # scaleds.append(scaled)
    reframed = series_to_supervised(scaled, 2, 1)
    print(reframed)
    reframed.drop(reframed.columns[[9,10,11]], axis=1, inplace=True)
    values = reframed.values
    print(reframed.head())
    train_size = int(len(dataset) * 0.7)  # 70%作为训练集
    train = values[:train_size, :]  # 矩阵前面是行，后面是列 所以是取所有列
    test = values[train_size:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]  # 最后一列作为输出标签
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    # 其中samples是训练序列的数目，timesteps是序列长度，features是每个序列的特征数目。
    train_X = train_X.reshape((train_X.shape[0], 2, 4))  # 进行维度变换
    test_X = test_X.reshape((test_X.shape[0], 2, 4))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)  # .shape 查看数据的维数
    # print(train_X)

    # 模型建立
    model = Sequential()
    model.add(GRU(20, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, kernel_regularizer=l2(0.001),
              recurrent_regularizer=l2(0.005)))
    model.add(GRU(20, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.005)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=600, batch_size=384, validation_data=(test_X, test_y)) #调这个？

    # make the prediction,为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
    yHat = model.predict(test_X)
    y = model.predict(train_X)
    test_X = test_X.reshape((test_X.shape[0], 8))

    '''
        这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
    '''
    
    print(test_X[:, -3:].shape[0],test_X[:, -3:].shape[1])
    yHat = yHat.reshape((len(yHat), 1))
    print(yHat.shape[0],yHat.shape[1])
    inv_yHat = concatenate((yHat, test_X[:, -3:]), axis=1)  # 数组拼接
    inv_yHat = scaler.inverse_transform(inv_yHat)
    inv_yHat = inv_yHat[:, 0]
    inv_yHats.append(inv_yHat)

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -3:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)  # 将标准化的数据转化为原来的范围
    inv_y = inv_y[:, 0]
    inv_ys.append(inv_y)

  inv_yHats = np.array(inv_yHats)
  inv_yHats = np.sum(inv_yHats, axis=0)
  inv_ys = np.array(inv_ys)
  inv_ys = np.sum(inv_ys, axis=0)
  rmse = sqrt(mean_squared_error(inv_yHats, inv_ys))
  print('Test RMSE: %.3f' % rmse)
  MAE = mae_value(inv_ys, inv_yHats)
  print('Test MAE: %.3f' % MAE)
  #x = np.arange(618,881, 1)
  plt.plot(inv_yHats, label='prediction value',color='#FF6666')
  plt.plot(inv_ys, label='detection value',color='#6699FF')
  #plt.plot(x,inv_yHats, label='prediction value')
  #plt.plot(x,inv_ys, label='detection value')
  plt.xlabel('Time(hour)')
  #plt.xlabel('Time(Unit:hour)')
  plt.ylabel('CO2 Concentration(ppm)')
  #  plt.ylabel('Ammonia Concentration(Unit:ppm)')
  #plt.title('Test set data prediction results of EEMD-GRU')
  plt.title('EEMD-GRU model of carbon dioxide value in winter') 
  plt.legend()
  #plt.savefig('./Test result of EEMD-GRU.jpg')  # 保存结果图到本地，必须写在plt.show()前面
  plt.show()
