import tensorflow_datasets as tfds
import pandas as pandas

data, info = tfds.load("wine_quality", with_info=True)
df = tfds.as_dataframe(data['train'])
print(info)
print(df.columns)
