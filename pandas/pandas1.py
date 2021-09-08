import tensorflow_datasets as tfds
import pandas as pandas

data, info = tfds.load("wine_quality", with_info=True)
df = tfds.as_dataframe(data['train'])
print(info)
print(df.columns)

#__________________________________________________

df2 = df.iloc[:, :11]
print(type(df2))
print(df2.shape)

features = df.iloc[:, :11]
label = df.iloc[:, 11:]

#print(label.shape)

#print(features[['features/alcohol']].describe())
#dx = features['features/alcohol'].plot.hist(bins=12, alpha=0.5)

print(label.describe())
lx = label.plot.hist(bins=12, alpha=0.5)

#__________________________________________________

df2 = df.query("quality == 6")[['features/alcohol']]
#print(df2)
print(type(df2))
print(df2.shape)
print(df2.describe())

df3 = df.query("quality == 9")[['features/alcohol']]
#print(df3)
print(type(df3))
print(df3.shape)
print(df3.describe())
