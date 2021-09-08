import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

data, info = tfds.load("wine_quality", with_info=True)
df = tfds.as_dataframe(data['train'])
print(info)
print(df.columns)

df_train = df.sample(frac=0.85, random_state=25)
df_test = df.drop(df_train.index)
print(df_train.shape, ' --  ', df_test.shape)

features_train = df_train.iloc[:, :11]
label_train = df_train.iloc[:, 11:]

print(features_train.shape)
print(features_train.dtypes)
print(features_train.isna().sum())

features_test = df_test.iloc[:, :11]
label_test = df_test.iloc[:, 11:]

print(features_test.shape)
print(features_test.dtypes)
print(features_test.isna().sum())

print(label_train.describe())
label_train.hist()

label_train = tf.keras.utils.to_categorical(label_train, NUM_CLASSES)
label_test = tf.keras.utils.to_categorical(label_test, NUM_CLASSES)

BATCH_SIZE = 250
NUM_INPUTS = features_train.shape[1]
NUM_CLASSES = 10
EPOCHS = 250
N_HIDDEN1 = 25000
N_HIDDEN2 = 4500
VALIDATION_SPLIT=0.1 
DROPOUT = 0.2

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(N_HIDDEN1, input_shape=(NUM_INPUTS,), name='dense_layer', activation='relu'))
model.add(tf.keras.layers.Dropout(DROPOUT))
model.add(tf.keras.layers.Dense(N_HIDDEN2,  name='dense_layer2', activation='relu'))
model.add(tf.keras.layers.Dropout(DROPOUT))
model.add(tf.keras.layers.Dense(NUM_CLASSES,  name='dense_layer3', activation='softmax'))

model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(features_train, label_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, validation_split=VALIDATION_SPLIT)

test_loss, test_acc  = model.evaluate(features_test, label_test)
print(test_loss, '  -  ', test_acc)