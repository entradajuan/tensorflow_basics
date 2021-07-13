import tensorflow as tf

message = tf.constant("This the hell gate!!!")

print(message)
print(type(message))

v_1 = tf.constant([1,2,3,4])
print(v_1)
print(type(v_1))


v_2 = tf.constant([1,2,3,4])
v_add = tf.add(v_1, v_2)

print(v_add)
print("hello, {}".format(v_add))
print(type(v_add))

