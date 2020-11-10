import sys
import time
import numpy as np
import pyarrow
from PyShmemML import shmem_ml_init, shmem_ml_finalize, PyShmemML1DD, PyShmemML2DD, rand, pe, npes, SGDRegressor
import sklearn.linear_model

# In client-server mode, only a single PE runs the program. It farms work out to the other PEs
assert pe() == 0

nsamples = 500000
vec = PyShmemML1DD(nsamples)
mat = PyShmemML2DD(nsamples, 5)

vec.clear(0)

print('Success')
