import tensorflow as tf
tf.keras.backend.clear_session()

# 1 .- GET DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2 .- PREPROCESS DATA
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 3.- BUILD MODEL
inputs = tf.keras.Input(shape=(784,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x2 = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x2)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# 4 .- TRAIN MODEL
print(type(x_train), 'size = ', x_train.shape)
print(type(y_train), 'size = ', y_train.shape)
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)

# 5 .- EVALUATE MODEL
print(type(x_test), 'size = ', x_test.shape)
print(type(y_test), 'size = ', y_test.shape)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# 6 .- SAVE MODEL
model.save('path_to_my_model.h5')
del model
# Recrea el mismo modelo, desde el archivo:
model2 = tf.keras.models.load_model('path_to_my_model.h5')
print(model2.summary())

