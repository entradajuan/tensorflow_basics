import tensorflow as tf

tf.keras.backend.clear_session()


inputs = tf.keras.Input(shape=(784,))

img_inputs = tf.keras.Input(shape=(32, 32, 3))

print(inputs.shape)
print(inputs.dtype)

dense = tf.keras.layers.Dense(64, activation='relu')
x = dense(inputs)

print(x)

x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

print(outputs)
