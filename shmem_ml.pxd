import numpy as np
# from numpy cimport int64_t
import pyarrow
from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow cimport CArray

np.import_array()

# cdef extern from "shmem_ml.cpp":
#     pass

cdef extern from "stdint.h":
    ctypedef signed int int64_t

cdef extern from "shmem_ml.hpp":
    cpdef void shmem_ml_init()
    cpdef void shmem_ml_finalize()

    cdef cppclass ShmemML1D[T]:
        ShmemML1D(int64_t) except +
        void clear(T clear_to)
        T get(int64_t)
        void sync()
        int64_t N()
        int64_t local_slice_start()
        int64_t local_slice_end()
        shared_ptr[CArray] get_arrow_array()
        void update_from_arrow_array(shared_ptr[CArray] src)

cdef extern from "shmem.h":
    cdef int shmem_my_pe()
    cdef int shmem_n_pes()
