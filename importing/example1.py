import pandas as pd
import tensorflow as tf

DATASET_PATH = '/content/drive/My Drive/Machine Learning/datos/BERT_sentiment_IMDB_Dataset.csv'

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(DATASET_PATH)
df = df[0:10000]

print(df)
print(type(df))
print(df.head())
print(df.isna().sum())
print(df.describe())
