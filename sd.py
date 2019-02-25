import numpy as np
v = np.array([[1,2,3],[4,5,6]])
u = np.array([[1,2,3],[6,4,4]])
index = v>u
print(v[index])
