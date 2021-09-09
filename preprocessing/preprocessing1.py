import pandas as pd
import numpy as np

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.1, 'class2'],
                   ['blue', 'XL', 15.3, 'class1'],
                   ])

df.columns = ['color', 'size', 'price', 'label']
print(df.head(), '\n')

size_mapping = {'XL': 1, 'L': 2, 'M': 3, 'S': 4 }
#df['size'] = df['size'].map(size_maping) 

size_change = lambda s : size_maping[s]
#df['size'] = df['size'].apply(size_change)
df['size'] = df['size'].map(size_change)

print(df.head(), '\n')

#class_mapping = {'class1': 0, 'class2':1}
class_mapping = {c:i-1 for i, c in enumerate(df['label'])}
ids_2_label = np.asarray(df['label'])
df['label'] = df['label'].map(class_mapping)

print(ids_2_label[1], '\n')
print('-- Labels --\n', df['label'].unique, '\n')
print(df.head(), '\n')

features = df.iloc[:, :3]
print(features)
print(type(features))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

features['color'] = LabelEncoder().fit_transform(features['color'])
print(features)

ohe = OneHotEncoder([0])
#ohe.fit_transform(features).toarray()



