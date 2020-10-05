import numpy as np
# from numpy cimport int64_t
import pyarrow
from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow cimport CArray, CRecordBatch, CTable

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

    cdef cppclass ReplicatedShmemML1D[T]:
        ReplicatedShmemML1D(int64_t) except +
        void reduce_all_sum()
        void bcast(int src_rank)
        shared_ptr[CArray] get_arrow_array()
        void update_from_arrow_array(shared_ptr[CArray] src)

    cdef cppclass ShmemML2D:
        ShmemML2D(int64_t, int64_t) except +
        int64_t M()
        int64_t N()
        int64_t rows_per_pe()
        shared_ptr[CTable] get_arrow_table()
        void update_from_arrow_table(shared_ptr[CTable] src)
        float get(int64_t row, int64_t col)


cdef extern from "shmem.h":
    cdef int shmem_my_pe()
    cdef int shmem_n_pes()
