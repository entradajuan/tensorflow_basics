import tensorflow as tf

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
#data = load_data()

# 2 .- PREPROCESS DATA



