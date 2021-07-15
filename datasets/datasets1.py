import tensorflow as tf
import tensorflow_datasets as tfds

print('\n 1__________________________________________________')

# See all registered datasets
builders = tfds.list_builders()
print (builders)

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']

print(info)

print('\n 2__________________________________________________')

#print(len(builders))
#for data in builders:
#  print(data)

dataset, info = tfds.load("ag_news_subset", with_info=True)
print(dataset)
print(info)
print(dataset['train'])
print(type(dataset['train']))

df1 = tfds.as_dataframe(dataset['train'].take(4), info)
print(df1)
print(type(df1))


print('\n 3__________________________________________________')

builder = tfds.builder('imdb_reviews')
builder.download_and_prepare()


datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_dataset = datasets['train']
train_dataset = train_dataset.batch(5).shuffle(50).take(2)

for data in train_dataset:
    print(data)

print('\n 4__________________________________________________')

dataset, info = tfds.load("blimp", with_info=True)
print(dataset)
print(info)
print(dataset['train'])
#print(type(dataset['train']))

tfds.as_dataframe(dataset['train'].take(4), info)
