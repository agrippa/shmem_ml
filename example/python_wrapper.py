from PyShmemML import shmem_ml_init, shmem_ml_finalize, PyShmemML1DD
# from shmem_ml import ml
# from shmem_ml import random

shmem_ml_init()

vec = PyShmemML1DD(10)
vec.clear(0)
vec.sync()
value_at_zero = vec.get(0)

print('Hello! ' + str(value_at_zero))

vec.get_local_arrow_array()

shmem_ml_finalize()

# ml.train_ml_model()

# random.rand(vec)
