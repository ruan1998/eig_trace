import pandas as pd

df = pd.read_csv('feature_points.csv')
# df = df.iloc[:,1:]
# df.time = range(63 * 96)
# df.time = df.time.apply(lambda t: t // 63)
# df .to_csv('feature_points.csv')
# df.to_csv('feature_points.csv',index=None)
print(df['IM_'])
