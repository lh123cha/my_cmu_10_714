import numpy as np

a = np.array([[2,-2,1],[-1,1,-1],[-3,-1,1]])
print(np.dot(a.transpose(),a))
q,r = np.linalg.qr(a)
print("q:",q)
print("r",r)