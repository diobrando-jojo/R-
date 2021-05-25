# -*- coding = utf-8 -*-
# @Time :  15:40
# @Author : cjj
# @File : risk_analyz_var.py
# @Software : PyCharm

import pandas as pd
import numpy as np
from sklearn import svm
import time

#读取现有CSV    (4708,8)
df = pd.read_csv('./SH600519.csv')

#修改列名
new_col = ['stock_code', 'date','open_p','max_p','min_p','close_p','volume_s','volume_p']
df.columns = new_col

#设定可迭代对象
iter_list = df['date'].str.split(':').tolist()
iter_list1 = list(set([i[0] for i in iter_list]))
#增加新列（昨日，上周，上月）交易量
df_v1 = df['volume_s'].shift(1)
df = pd.concat([df,df_v1],axis = 1)
new_col.append('volume_s1')
df.columns = new_col

df_v7 = df['volume_s'].shift(7)
df = pd.concat([df,df_v7],axis = 1)
new_col.append('volume_s7')
df.columns = new_col

df_v30 = df['volume_s'].shift(30)
df = pd.concat([df,df_v30],axis = 1)
new_col.append('volume_s30')
df.columns = new_col

#振幅，收益率

df.eval('gap = max_p - min_p', inplace=True)
df.eval('re = (close_p - open_p)/open_p',inplace=True)

#OBV指标计算
for i in range(df.shape[0]):
    if i == 0:
        df.loc[i,'OBV'] = df.loc[i,'volume_s']
    elif i > 0:
        if df.loc[i,'close_p'] > df.loc[i-1,'close_p']:
            df.loc[i, 'OBV'] = df.loc[i-1,'OBV'] + df.loc[i,'volume_s']
        elif df.loc[i,'close_p'] == df.loc[i-1,'close_p']:
            df.loc[i, 'OBV'] = df.loc[i - 1, 'OBV']
        elif df.loc[i,'close_p'] < df.loc[i-1,'close_p']:
            df.loc[i, 'OBV'] = df.loc[i-1,'OBV'] - df.loc[i,'volume_s']

#KDJ指标计算
low_list = df['min_p'].rolling(9, min_periods=9).min()
low_list.fillna(value = df['min_p'].expanding().min(), inplace = True)
high_list = df['max_p'].rolling(9, min_periods=9).max()
high_list.fillna(value = df['max_p'].expanding().max(), inplace = True)
rsv = (df['close_p'] - low_list) / (high_list - low_list) * 100

df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
df['D'] = df['K'].ewm(com=2).mean()
df['J'] = 3 * df['K'] - 2 * df['D']

#CCI(7日）
# CCI（N日）=（TP－MA）÷MD÷0.015
# 其中，TP=（最高价+最低价+收盘价）÷3
# MA=最近N日收盘价的累计之和÷N
# MD=最近N日（MA－收盘价）的累计之和÷N
# 0.015为计算系数，N为计算周期
df['MA'] = df['close_p'].rolling(7).mean()
df['MD'] = df['MA'].rolling(7).mean() - df['close_p'].rolling(7).mean()
df['CCI'] = ((df['max_p'] + df['min_p'] + df['close_p'])/3 - df['MA'])/df['MD']/0.015

#MACD
def get_ema(data,days):
    for i in range(len(data)):
        if i == 0:
            data.loc[i, 'EMA' + str(days)] = data.loc[i, 'close_p']
        elif i>0 :
            data.loc[i, 'EMA' + str(days)] = (2 * data.loc[i, 'close_p'] + (days - 1) * data.loc[i-1, 'EMA' + str(days)])/(days + 1)
    return data

get_ema(df,9)
get_ema(df,12)
get_ema(df,26)

df.loc[:,'MACD'] = 224/15*df.loc[:,'EMA9'] - 16/3 * df.loc[:,'EMA12'] + 16/17*df.loc[:,'EMA26']
#VaR,排序法,a = 75%

for i in range(len(df)):
    if i >= 253:
        re_list = np.array(df.loc[i-252:i, 're']).tolist()
        re_list.sort()
        n, a = len(re_list), 0.75
        var_point = int(n * (1 - a))
        var_value = (re_list[var_point] + re_list[var_point+1]) / 2
        df.loc[i,'var'] = var_value
    else:
        df.loc[i, 'var'] = 0

#数据清洗，删除空行
df.dropna(subset=['volume_s30'],inplace=True)
df = df.reset_index(drop = True)

#截取一年和需要的数据
#截取2020513开始的数据到2021513，删除没必要的列
open_day = '2019-01-01'
df_1 = df['date'] >= open_day
df = df[df_1]
df = df.drop(['EMA12','EMA9','EMA26','MA','MD'],axis=1)
df = df.reset_index(drop = True)



#添加新列，用于标记股票的违约情况
for i in range(len(df)):
    if df.loc[i,'re'] <= df.loc[i,'var']:
        df.loc[i,'credit'] = -1
    else:
        df.loc[i, 'credit'] = 1

#保存一下
df.to_csv('C:/Users/oscar/Desktop/R语言大作业/clean_stock_data.csv')

#引入支持向量机预测



