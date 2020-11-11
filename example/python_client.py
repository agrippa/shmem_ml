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

vec.sync()

value_at_zero = vec.get(0)
assert value_at_zero == 0.
vec.sync()

print('PE=' + str(pe()) + ' N=' + str(vec.N()) + ' local slice start=' +
        str(vec.local_slice_start()) + ' local slice end=' +
        str(vec.local_slice_end()))

vec = rand(vec)
mat = rand(mat)

vec = vec.apply(lambda i, x, vec: i / vec.N())
mat = mat.apply(lambda i, j, x, mat: i / mat.M())

vec.sync()
mat.sync()

# Validate that the applys above did what we wanted them to do.
for i in range(vec.N()):
    assert vec.get(i) == i / vec.N(), (i, vec.get(i), vec.get(i+1))

for i in range(mat.M()):
    for j in range(mat.N()):
        assert mat.get(i, j) == i / mat.M()

print('Success')
