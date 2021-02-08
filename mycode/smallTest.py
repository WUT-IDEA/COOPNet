# -*- coding:utf-8 -*-

'''
word embedding测试
在GTX960上，18s一轮
经过30轮迭代，训练集准确率为98.41%，测试集准确率为89.03%
Dropout不能用太多，否则信息损失太严重
'''

import numpy as np
import pandas as pd
import jieba


arr = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]])
df = pd.DataFrame(arr)
print(df)

df = df.iloc[np.random.permutation(len(df))]
print(df)

df.columns = ['a', 'b', 'c']
print(df)

se = df['a']
print(se[0])
print(se.iloc[0])
