import tensorflow as tf

tf.keras.backend.clear_session()

inputs = tf.keras.Input(shape=(784,))

img_inputs = tf.keras.Input(shape=(32, 32, 3))

print(inputs.shape)
print(inputs.dtype)

dense = tf.keras.layers.Dense(64, activation='relu')
x = dense(inputs)
print(x)

x2 = tf.keras.layers.Dense(128, activation='relu')(x)
print(x2)

x3 = tf.keras.layers.Dense(32, activation='relu')(x)
print(x3)

outputs = tf.keras.layers.Dense(10, activation='softmax')(x3)
print(outputs)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
