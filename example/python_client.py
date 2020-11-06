import sys
import time
import numpy as np
import pyarrow
from PyShmemML import shmem_ml_init, shmem_ml_finalize, PyShmemML1DD, PyShmemML2DD, rand, pe, npes, SGDRegressor
import sklearn.linear_model

nsamples = 500000
vec = PyShmemML1DD(nsamples)

if pe() == 0:
    print('Success')
