import numpy as np
import pandas as pd

# 创建DataFrame，包含缺失值
df = pd.DataFrame({'A': [0, 1, 2, np.nan, 4, 5], 'B': [5, np.nan, 7, 8, 9, 10]})
print(df)

# # 线性插值
# df_interpolated = df.interpolate()
#
# print(df_interpolated)

df['B'] = df['B'].interpolate()
print(df)
