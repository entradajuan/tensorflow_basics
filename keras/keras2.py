import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

import pandas as pd
import numpy as np

## PREPROCESS _________________

data = pd.read_csv('importing/spam.csv', encoding = "ISO-8859-1")
data = data[['v1','v2']] 

#conver = lambda s : int(s=='ham') 

data['v1'] = data['v1'].apply(conver)
#print(data[['v1','v2']].head())
#print(type(data))
#print(data.describe())

label = data['v1']
sentences = data['v2']
#print(sentences)
print(type(sentences))

## VECTORIZE
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
text_sequences = tokenizer.texts_to_sequences(sentences)

text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences)
print(len(text_sequences[0]))
print(len(text_sequences[5571]))

## MODEL
N_HIDDEN = 25
OUTPUT_BITS = 1
DROPOUT = 0.2
EPOCHS = 1
BATCH_SIZE = 50

#build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
   		input_shape=(len(text_sequences[0]),),
   		name='dense_layer', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
   		name='dense_layer_2', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(OUTPUT_BITS,
   		name='dense_layer_3', activation='softmax'))



#model = tf.keras.models.Sequential()
#model.add(layers.Dense(128, input_shape=(189,), activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(1, activation='sigmoid'))

# summary of the model
model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

#label = np.asarray(label).astype('float32').reshape((-1,1))
#X_test, y_test = 
#score = model.fit(text_sequences, label, epochs= EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_test, y_test))
score = model.fit(text_sequences, label, epochs= EPOCHS, batch_size = BATCH_SIZE)
