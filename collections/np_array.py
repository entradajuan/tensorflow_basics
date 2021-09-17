import numpy as numpy


arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)

suma = lambda x : x + 3
vfunc = np.vectorize(suma)
vfunc(arr)

print(arr)

# ____________________________________________________________


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)

newarr = newarr.flatten()
print(newarr)

# ____________________________________________________________


arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)

# ____________________________________________________________


texts = ([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.1, 'class2'],
                   ['blue', 'XL', 15.3, 'class1'],
                   ])



arr
