import tensorflow as tf

digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])

print(digits)
print(type(digits))

print(digits[0])
print(type(digits[0]))


text = ["Jeff Bezos’s Blue Origin gets OK from the FAA to launch him and 3 others into space",
        "Schumer, other Democrats to unveil draft bill for cannabis decriminalization on Wednesday",
        "Identity thief used burner phones and Apple Pay to buy diamond-encrusted bitcoin medallion — and $600K in luxury goods",]

dataset = tf.ragged.constant(text)
print(dataset)
print(type(dataset))

text = [["Jeff Bezos’s Blue Origin gets OK from the FAA to launch him and 3 others into space"],
        ["Schumer, other Democrats to unveil draft bill for cannabis decriminalization on Wednesday"],
        ["Identity thief used burner phones and Apple Pay to buy diamond-encrusted bitcoin medallion — and $600K in luxury goods"]]

dataset = tf.ragged.constant(text)
print(dataset)
print(type(dataset))
print(dataset[0])
print(type(dataset[0]))