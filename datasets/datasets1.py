import tensorflow as tf
import tensorflow_datasets as tfds

# See all registered datasets
builders = tfds.list_builders()
print (builders)

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']

print(info)

print('\n__________________________________________________')

#print(len(builders))
#for data in builders:
#  print(data)

dataset, info = tfds.load("ag_news_subset", with_info=True)
print(dataset)
print(info)
print(dataset['train'])
print(type(dataset['train']))

tfds.as_dataframe(dataset['train'].take(4), info)

print('\n__________________________________________________')

builder = tfds.builder('blimp')
builder.download_and_prepare()
ds = builder.as_dataset(split='train', shuffle_files=True)
print(ds)


builder = tfds.builder('blimp')
#builder.download_and_prepare()
ds = builder.as_dataset(split='train', shuffle_files=True)
print(type(ds))
print(ds)

##print(ds['UID'], info) ????????????