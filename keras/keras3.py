import tensorflow as tf
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')


def setupGPU():
    ######## GPU CONFIGS FOR RTX 2070 ###############
    ## Please ignore if not training on GPU       ##
    ## this is important for running CuDNN on GPU ##

    tf.keras.backend.clear_session() #- for easy reset of notebook state

    # chck if GPU can be seen by TF
    tf.config.list_physical_devices('GPU')
    #tf.debugging.set_log_device_placement(True)  # only to check GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    ###############################################


def load_data():
  DATASET_PATH = '/content/drive/My Drive/Machine Learning/datos/BERT_sentiment_IMDB_Dataset.csv'
  data = pd.read_csv(DATASET_PATH)

  return data

# GPU
setupGPU()

# 1 .- LOAD DATA

data = load_data()
print(data)
#print(data.describe())
#print(data.sentiment.describe())

# 2 .- PREPROCESS DATA
training_data = data.sample(frac=0.8, random_state=25)
testing_data = data.drop(training_data.index)

train_senteces = training_data['review']
train_labels =  training_data['sentiment']

# Tokenizer 
def get_tokenizer(data):
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(data)

  return tokenizer

tokenizer = get_tokenizer(data['review'])

sentences = tokenizer.texts_to_sequences(train_senteces)


