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

# To numpy or Series and PREPROCESSING and off to feed the model to be trained
text = df['review']
label = df['sentiment']

print(text)
print(type(text))

## PANDAS SERIES VS NUMPY ARRAY?
## WHY DO WE NEED TENSORFLOW???


