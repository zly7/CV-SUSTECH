import numpy as np
a = [[1,2],[2,3],[3,4]]
temp = a[:2]
print(temp)
temp[0][1]=99 
print(a)

a = [[1,2],[2,3],[3,4]]
temp = a[:2]
print(temp)
temp[0]=99 
print(a)


a = np.array([[1,2],[2,3],[3,4]])
temp = a[:2]
temp[0][1]=99 
print(a)

a = np.array([[1,2],[2,3],[3,4]])
temp = a[:2]
temp[0]=np.array([99,99]) 
print(a)