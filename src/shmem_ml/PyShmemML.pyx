# distutils: language = c++
#cython: language_level=3
import pyarrow
from pyarrow.lib cimport *

from shmem_ml cimport shmem_ml_init as c_shmem_ml_init
from shmem_ml cimport shmem_ml_finalize as c_shmem_ml_finalize

import numpy as np
cimport numpy as np
from numpy cimport int64_t

from shmem_ml cimport ShmemML1D

np.import_array()

def shmem_ml_init():
    c_shmem_ml_init()

def shmem_ml_finalize():
    c_shmem_ml_finalize()


cdef class PyShmemML1DD:
    cdef ShmemML1D[double]* c_vec

    def __cinit__(self, int64_t N):
        self.c_vec = new ShmemML1D[double](N)

    def __dealloc__(self):
        del self.c_vec

    def clear(self, double clear_to):
        self.c_vec.clear(clear_to)

    def get(self, int64_t global_index):
        return self.c_vec.get(global_index)

    def sync(self):
        self.c_vec.sync()

    def get_local_arrow_array(self):
        return pyarrow_wrap_array(self.c_vec.get_arrow_array())
