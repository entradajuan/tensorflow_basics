import tensorflow as tf
import tensorflow_datasets as tfds

print(tfds.list_builders())
data, info = tfds.load("anli", with_info=True)
print(data['train'])
train_dataset = data['train']
train_dataset = train_dataset.batch(5).shuffle(50).take(2)
print('\n\n')

print(train_dataset)
print(type(train_dataset))
print(tfds.as_dataframe(data['train'].take(4), info))

