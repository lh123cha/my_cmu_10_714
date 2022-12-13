import numpy as np
from numpy import linalg as la
shape = (4,4,4)
def compact_strides(shape):
        """ Utility function to compute compact strides """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])
print(compact_strides(shape))