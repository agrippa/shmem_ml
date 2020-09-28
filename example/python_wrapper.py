import time
import numpy as np
import pyarrow
from PyShmemML import shmem_ml_init, shmem_ml_finalize, PyShmemML1DD, PyShmemML2DD, rand, pe, npes, SGDRegressor

# Set up runtime
shmem_ml_init()

start_time = time.time()

# Distributed, collective allocation of a 10-element array of float64
vec = PyShmemML1DD(10)
mat = PyShmemML2DD(10, 5)

# Clear all values in vec to 0.0
vec.clear(0)

# Global barrier
vec.sync()

# Every rank uses RMA to grab the 0-th element of vec
value_at_zero = vec.get(0)
assert value_at_zero == 0.

print('PE=' + str(pe()) + ' N=' + str(vec.N()) + ' local slice start=' +
        str(vec.local_slice_start()) + ' local slice end=' +
        str(vec.local_slice_end()))

# Initialize vec with random values (in place)
vec = rand(vec)

clf = SGDRegressor(max_iter=10)
clf.fit(mat, vec)
pred = clf.transform(mat)

elapsed_time = time.time() - start_time
print('PE ' + str(pe()) + ' elapsed: ' + str(elapsed_time) + ' s')

# Shut down runtime
shmem_ml_finalize()

# ml.train_ml_model()

