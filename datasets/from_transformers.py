from datasets import list_datasets, load_dataset, list_metrics, load_metric

# Check https://huggingface.co/datasets
print(list_datasets())
print(len(list_datasets()))

#downloaded_dataset = load_dataset('squad')
#downloaded_dataset = load_dataset('ag_news')
downloaded_dataset = load_dataset("cnn_dailymail", '3.0.0')

print(downloaded_dataset)
train_data = downloaded_dataset['train'] 
print(train_data)
print(type(train_data))
print(train_data[0])

