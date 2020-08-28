# distutils: language = c++
#cython: language_level=3
import pyarrow
from pyarrow.lib cimport *

import numpy as np
# cimport numpy as np
# from numpy cimport int64_t

# np.import_array()

from shmem_ml cimport shmem_ml_init as c_shmem_ml_init
from shmem_ml cimport shmem_ml_finalize as c_shmem_ml_finalize
from shmem_ml cimport shmem_my_pe as c_shmem_my_pe
from shmem_ml cimport shmem_n_pes as c_shmem_n_pes

from shmem_ml cimport ShmemML1D


def shmem_ml_init():
    c_shmem_ml_init()
    tmp = pyarrow.array([])

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

    def N(self):
        return self.c_vec.N()

    def local_slice_start(self):
        return self.c_vec.local_slice_start()

    def local_slice_end(self):
        return self.c_vec.local_slice_end()

    def get_local_arrow_array(self):
        return pyarrow_wrap_array(self.c_vec.get_arrow_array())

    def update_from_arrow(self, arrow_arr):
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(arrow_arr)
        self.c_vec.update_from_arrow_array(arr)


def rand(vec):
    assert isinstance(vec, PyShmemML1DD)
    arr = vec.get_local_arrow_array()
    np_arr = arr.to_numpy(zero_copy_only=False, writable=True)
    np_arr[:] = np.random.rand(np_arr.shape[0])
    pyarr = pyarrow.array(np_arr)
    vec.update_from_arrow(pyarr)
    return vec

def pe():
    return c_shmem_my_pe()

def npes():
    return c_shmem_n_pes()
