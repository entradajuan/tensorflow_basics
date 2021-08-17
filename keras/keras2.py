import pandas as pd
import numpy as np
import tensorflow as tf

# 1 .- LOAD DATA
data = pd.read_csv('importing/spam.csv', encoding = "ISO-8859-1")
data = data[['v1', 'v2']]

# 2 .- PREPROCESS DATA
convert = lambda s : int(s=='ham')
data['v1'] = data['v1'].apply(convert)

training_data = data.sample(frac=0.8, random_state=25)
testing_data = data.drop(training_data.index)

sentences = training_data['v2']
labels = training_data['v1']

print(sentences[3])

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['v2'])
sentences = tokenizer.texts_to_sequences(sentences)

print(sentences[3])
print(type(sentences))

MAX_LEN = 200
sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=MAX_LEN)

print(sentences[3])
print(type(sentences))

# 3 .- BUILD MODEL
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

model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

# 4 .- BUILD MODEL
score = model.fit(sentences, labels, epochs= EPOCHS, batch_size = BATCH_SIZE)

# 5 .- EVALUATE MODEL
eval_sentences = testing_data['v2']
eval_labels = testing_data['v1']

eval_sentences = tokenizer.texts_to_sequences(eval_sentences)
eval_sentences = tf.keras.preprocessing.sequence.pad_sequences(eval_sentences, maxlen=MAX_LEN)

score = model.evaluate(eval_sentences, eval_labels, batch_size=BATCH_SIZE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# 6 .- USE MODEL TO PREDICT
#text = ['Hi Juani, I will need an answer tomorrow in the morning. regards']
text = ['Hi, Get the new iPhone in this link and get a 50% saving!!. Best regards']

sentence = tokenizer.texts_to_sequences(text)
print(sentence)
sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, maxlen=MAX_LEN)
print(sentence)

print(type(sentence))
prediction = model.predict(sentence)
print(prediction)



