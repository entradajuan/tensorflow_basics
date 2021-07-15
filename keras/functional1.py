import tensorflow as tf

tf.keras.backend.clear_session()

inputs = tf.keras.Input(shape=(784,))

#img_inputs = tf.keras.Input(shape=(32, 32, 3))
#print(inputs.shape)
#print(inputs.dtype)

dense = tf.keras.layers.Dense(64, activation='relu')
x = dense(inputs)
print(x)

x2 = tf.keras.layers.Dense(128, activation='relu')(x)
print(x2)

x3 = tf.keras.layers.Dense(32, activation='relu')(x)
print(x3)

outputs = tf.keras.layers.Dense(10, activation='softmax')(x3)
print(outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

print(tf.keras.utils.plot_model(model, 'my_first_model.png'))
print(tf.keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True))

## TRAINING 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# Save and Load
model.save('path_to_my_model.h5')
del model
# Recrea el mismo modelo, desde el archivo:
model2 = keras.models.load_model('path_to_my_model.h5')
print(model2.summary())