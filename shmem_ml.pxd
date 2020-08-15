from numpy cimport int64_t
import pyarrow
from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow cimport CArray

cdef extern from "shmem_ml.hpp":
    cpdef void shmem_ml_init()
    cpdef void shmem_ml_finalize()

    cdef cppclass ShmemML1D[T]:
        ShmemML1D(int64_t) except +
        void clear(T clear_to)
        T get(int64_t)
        void sync()
        shared_ptr[CArray] get_arrow_array()