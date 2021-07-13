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

text = ["Jeff Bezos’s Blue Origin gets OK from the FAA to launch him and 3 others into space",
        "Schumer, other Democrats to unveil draft bill for cannabis decriminalization on Wednesday",
        "Identity thief used burner phones and Apple Pay to buy diamond-encrusted bitcoin medallion — and $600K in luxury goods",]

dataset = tf.constant(text)
print(dataset)
print(type(dataset))

print("{}".format(dataset[0]))