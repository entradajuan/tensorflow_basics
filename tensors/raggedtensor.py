import math
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

data = tf.RaggedTensor.from_value_rowids(
    values=[3, 1, 4, 1, 5, 9, 2],
    value_rowids=[0, 0, 0, 0, 2, 2, 3])

print(data)

queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],
                              ['Pause'],
                              ['Will', 'it', 'rain', 'later', 'today']])


# Create an embedding table.
num_buckets = 1024
embedding_size = 4
embedding_table = tf.Variable(
    tf.random.truncated_normal([num_buckets, embedding_size],
                       stddev=1.0 / math.sqrt(embedding_size)))

# Look up the embedding for each word.
word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
word_embeddings = tf.nn.embedding_lookup(embedding_table, word_buckets)     

# Add markers to the beginning and end of each sentence.
marker = tf.fill([queries.nrows(), 1], '#')
padded = tf.concat([marker, queries, marker], axis=1)                       

# Build word bigrams and look up embeddings.
bigrams = tf.strings.join([padded[:, :-1], padded[:, 1:]], separator='+')   

bigram_buckets = tf.strings.to_hash_bucket_fast(bigrams, num_buckets)
bigram_embeddings = tf.nn.embedding_lookup(embedding_table, bigram_buckets) 

# Find the average embedding for each sentence
all_embeddings = tf.concat([word_embeddings, bigram_embeddings], axis=1)    
avg_embedding = tf.reduce_mean(all_embeddings, axis=1)                      
print(avg_embedding)