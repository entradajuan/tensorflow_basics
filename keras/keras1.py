import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

max_len = 200
n_words = 10000
dim_embedding = 256
EPOCHS = 20
BATCH_SIZE =500

## FUNCIONES _____________________________________

def load_data():
  # keras datasets
  (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)
  X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
  X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
  return (X_train, y_train), (X_test, y_test)

def build_model():
  model = models.Sequential()
  model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
  model.add(layers.Dropout(0.3))
  model.add(layers.GlobalMaxPooling1D())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1, activation='sigmoid'))
  return model


## INICIO _____________________________________

(X_train, y_train), (X_test, y_test) = load_data()
print(X_train[0])
print(y_train[0])

model=build_model()
model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

score = model.fit(X_train, y_train, epochs= EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

## PREDICTIONS!!!
print(type(X_train))
print(X_train.shape)

ej1 = X_train[:1]
print(ej1)
print(type(ej1))
print(ej1.shape)

print('\n\n')
res = model.predict(ej1)
print(res)
print(type(res))

print(model.predict_classes(ej1))
