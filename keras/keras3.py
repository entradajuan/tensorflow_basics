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
#_______________________________________________________________________________
# GPU
setupGPU()


#_______________________________________________________________________________
# 1 .- LOAD DATA
data = load_data()

#_______________________________________________________________________________
# 2 .- PREPROCESS DATA
convert = lambda s : int(s == 'positive')
data['sentiment'] = data['sentiment'].apply(convert)
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
MAX_LEN = 200
sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=MAX_LEN)




#_______________________________________________________________________________
# 3 .- BUILD MODEL
def get_model1():
  inputs = tf.keras.Input(shape=(200,))
  x = tf.keras.layers.Dense(64, activation='relu')(inputs)
  x2 = tf.keras.layers.Dense(32, activation='relu')(x)
  x3 = tf.keras.layers.Dense(2500, activation='relu')(x2)
  x4 = tf.keras.layers.Dense(250, activation='relu')(x3)
  outputs = tf.keras.layers.Dense(2, activation='softmax')(x4)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
  return model

def get_model2():
  N_HIDDEN = 250
  OUTPUT_BITS = 1
  DROPOUT = 0.2
  EPOCHS = 50
  BATCH_SIZE = 50
  INPUT_SHAPE = (len(sentences[3]), )
  print(INPUT_SHAPE)

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(N_HIDDEN,
   		input_shape=INPUT_SHAPE,
   		name='dense_layer_1', activation='relu'))
  model.add(tf.keras.layers.Dropout(DROPOUT))
  model.add(tf.keras.layers.Dense(N_HIDDEN,
   		name='dense_layer_2', activation='relu'))
  model.add(tf.keras.layers.Dropout(DROPOUT))
  model.add(tf.keras.layers.Dense(OUTPUT_BITS,
   		name='dense_layer_3', activation='softmax'))
#  model.summary()
  model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

  return model 

model = get_model2()


#_______________________________________________________________________________
# 4 .- TRAIN MODEL
history = model.fit(sentences, train_labels,
                    batch_size=128,
                    epochs=50,
                    validation_split=0.2)


#_______________________________________________________________________________
# 6 .- EVALUATE MODEL
test_senteces = testing_data['review']
test_labels =  testing_data['sentiment']
sentences = tokenizer.texts_to_sequences(test_senteces)
sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=MAX_LEN)
test_scores = model.evaluate(sentences, test_labels, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])


#_______________________________________________________________________________
# 7 .- TRY MODEL
sentences = ['The movie was awesome!!! I loved it!!!']
sentences = tokenizer.texts_to_sequences(sentences)
sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=MAX_LEN)
prediction = model.predict(sentences)
print(prediction)