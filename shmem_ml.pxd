import numpy as np
# from numpy cimport int64_t
import pyarrow
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from pyarrow.includes.libarrow cimport CArray, CRecordBatch, CTable

np.import_array()

cdef extern from "stdint.h":
    ctypedef signed int int64_t

cdef extern from "shmem_ml.hpp":
    cpdef void shmem_ml_init()
    cpdef void shmem_ml_finalize()
    cpdef bool is_client_server_mode()
    cpdef void end_cmd()
    cpdef void send_sgd_fit_cmd(unsigned x_id, unsigned y_id, char* s, int s_length)
    cpdef void send_sgd_predict_cmd(unsigned x_id, char* s, int s_length)
    cpdef void send_sequential_fit_cmd(unsigned x_id, unsigned y_id, char* s, int s_length)
    cpdef void send_sequential_predict_cmd(unsigned x_id, char* s, int s_length)

    cdef cppclass shmem_ml_command:
        pass

    # From shmem_ml_cmd.hpp
    cdef shmem_ml_command CREATE_1D
    cdef shmem_ml_command DESTROY_1D
    cdef shmem_ml_command CREATE_2D
    cdef shmem_ml_command DESTROY_2D
    cdef shmem_ml_command CLEAR_1D
    cdef shmem_ml_command SYNC_1D
    cdef shmem_ml_command GET_1D
    cdef shmem_ml_command RAND_1D
    cdef shmem_ml_command RAND_2D
    cdef shmem_ml_command APPLY_1D
    cdef shmem_ml_command APPLY_2D
    cdef shmem_ml_command SGD_FIT
    cdef shmem_ml_command SGD_PREDICT
    cdef shmem_ml_command SEQUENTIAL_FIT
    cdef shmem_ml_command SEQUENTIAL_PREDICT
    cdef shmem_ml_command CMD_DONE
    cdef shmem_ml_command CMD_INVALID

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
        void send_rand_1d_cmd()
        void send_apply_1d_cmd(char* s, int length)
        unsigned get_id()

    cdef cppclass shmem_ml_py_cmd:
        shmem_ml_command get_cmd()
        void* get_arr()
        char* get_str()
        int get_str_length()
        void *get_arr_2()

    cpdef shmem_ml_py_cmd command_loop()

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
        void sync()
        void send_rand_2d_cmd()
        void send_apply_2d_cmd(char* s, int length)
        unsigned get_id()


cdef extern from "shmem.h":
    cdef int shmem_my_pe()
    cdef int shmem_n_pes()
