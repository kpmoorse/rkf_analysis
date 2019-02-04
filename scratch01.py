import numpy as np


class Variable(np.ndarray):

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj.metadata = kwargs
        return obj

var = Variable([1,2,3,4], comment="testmeta")