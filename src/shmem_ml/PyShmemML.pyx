# distutils: language = c++
#cython: language_level=3
import pyarrow
from pyarrow.lib cimport *

import sys
import atexit
import numpy as np
import sklearn
import sklearn.linear_model
from sklearn.linear_model import SGDRegressor as SK_SGDRegressor

from shmem_ml cimport shmem_ml_init as c_shmem_ml_init
from shmem_ml cimport shmem_ml_finalize as c_shmem_ml_finalize
from shmem_ml cimport shmem_my_pe as c_shmem_my_pe
from shmem_ml cimport shmem_n_pes as c_shmem_n_pes

from shmem_ml cimport ShmemML1D, ReplicatedShmemML1D, ShmemML2D


def shmem_ml_finalize():
    c_shmem_ml_finalize()


def shmem_ml_init():
    c_shmem_ml_init()
    # Force initialization of pyarrow so that it doesn't happen in a performance
    # critical section.
    tmp = pyarrow.array([])

    # atexit.register(shmem_ml_finalize)


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


cdef class PyShmemML2DD:
    cdef ShmemML2D* c_mat

    def __cinit__(self, int64_t M, int64_t N):
        self.c_mat = new ShmemML2D(M, N)

    def __dealloc__(self):
        del self.c_mat

    def N(self):
        return self.c_mat.N()

    def M(self):
        return self.c_mat.M()

    def get_local_arrow_record_batch(self):
        return pyarrow_wrap_batch(self.c_mat.get_arrow_record_batch())

    def update_from_arrow(self, arrow_record_batch):
        cdef shared_ptr[CRecordBatch] batch = pyarrow_unwrap_batch(arrow_record_batch)
        self.c_mat.update_from_arrow_record_batch(batch)


cdef class PyReplicatedShmemML1DD:
    cdef ReplicatedShmemML1D[double]* c_vec

    def update_from_arrow(self, arrow_arr):
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(arrow_arr)
        self.c_vec.update_from_arrow_array(arr)

    def reduce_all_sum(self):
        return self.c_vec.reduce_all_sum()

    def get_local_arrow_array(self):
        return pyarrow_wrap_array(self.c_vec.get_arrow_array())


def rand(vec):
    assert isinstance(vec, PyShmemML1DD)
    arr = vec.get_local_arrow_array()
    np_arr = arr.to_numpy(zero_copy_only=False, writable=True)
    np_arr[:] = np.random.rand(np_arr.shape[0])
    pyarr = pyarrow.array(np_arr)
    vec.update_from_arrow(pyarr)
    return vec


class SGDRegressor:
    def __init__(self, max_iter=1, random_state=32):
        self.model = SK_SGDRegressor(max_iter=1, tol=None,
                random_state=random_state, learning_rate='constant')
        self.max_iter = max_iter

    def _copy_coef_intercept(self):
        save_coef = np.zeros(len(self.model.coef_))
        save_intercept = np.zeros(len(self.model.intercept_))
        for i in range(len(self.model.coef_)):
            save_coef[i] = self.model.coef_[i]
        for i in range(len(self.model.intercept_)):
            save_intercept[i] = self.model.intercept_[i]
        return save_coef, save_intercept

    def _compute_coef_intercept_grad(self, prev_coef, prev_intercept, after_coef, after_intercept):
        coef_grad = np.zeros(len(prev_coef))
        intercept_grad = np.zeros(len(prev_intercept))

        for i in range(len(prev_coef)):
            coef_grad[i] = after_coef[i] - prev_coef[i]
        for i in range(len(prev_intercept)):
            intercept_grad[i] = after_intercept[i] - prev_intercept[i]

        return coef_grad, intercept_grad

    def _update_coef_intercept(self, prev_coef, prev_intercept, grad_coef, grad_intercept):
        for i in range(len(self.model.coef_)):
            self.model.coef_[i] = prev_coef[i] + grad_coef[i]
        for i in range(len(self.model.intercept_)):
            self.model.intercept_[i] = prev_intercept[i] + grad_intercept[i]

    def fit(self, x, y):
        assert isinstance(x, PyShmemML2DD)
        assert isinstance(y, PyShmemML1DD)

        x_arr = x.get_local_arrow_record_batch()
        x_arr = x_arr.to_pandas(zero_copy_only=True)
        print('type of x_arr = ' + str(type(x_arr)))

        y_arr = y.get_local_arrow_array()
        y_arr = y_arr.to_numpy(zero_copy_only=True)

        # Initialize coefficients and intercepts on all ranks to the same seed
        self.model.fit(x_arr, y_arr);
        coef, intercept = self._copy_coef_intercept()
        arrow_coef = pyarrow.array(coef)
        arrow_intercept = pyarrow.array(intercept)
        dist_coef_grad = PyReplicatedShmemML1DD(len(arrow_coef))
        dist_intercept_grad = PyReplicatedShmemML1DD(len(arrow_intercept))
        dist_coef_grad.update_from_arrow(arrow_coef)
        dist_intercept_grad.update_from_arrow(arrow_intercept)
        dist_coef_grad.bcast(0)
        dist_intercept_grad.bcast(0)
        set_coef = dist_coef_grad.get_local_arrow_array().to_numpy(zero_copy_only=True)
        set_intercept = dist_intercept_grad.get_local_arrow_array().to_numpy(zero_copy_only=True)
        self._update_coef_intercept(set_coef, set_intercept, [0.0] * len(set_coef), [0.0] * len(set_intercept))

        for it in range(1, self.max_iter):
            prev_coef, prev_intercept = self._copy_coef_intercept()

            self.model.partial_fit(x_arr, y_arr)

            after_coef, after_intercept = self._copy_coef_intercept()
            coef_grad, intercept_grad = self._compute_coef_intercept_grad(
                    prev_coef, prev_intercept, after_coef, after_intercept)

            arrow_coef_grad = pyarrow.array(coef_grad)
            arrow_intercept_grad = pyarrow.array(intercept_grad)

            # Perform a global sum of coef_grad and intercept_grad over SHMEM,
            # then add to prev_coef and prev_intercept and reset coef and
            # intercept in the model to the value
            dist_coef_grad.update_from_arrow(arrow_coef_grad)
            dist_intercept_grad.update_from_arrow(arrow_intercept_grad)
            dist_coef_grad.reduce_all_sum()
            dist_intercept_grad.reduce_all_sum()

            all_coef_grads = dist_coef_grad.get_local_arrow_array().to_numpy(zero_copy_only=True)
            all_intercept_grads = dist_intercept_grad.get_local_arrow_array().to_numpy(zero_copy_only=True)
            self._update_coef_intercept(prev_coef, prev_intercept,
                    all_coef_grads, all_intercept_grads)


    def transform(self, x):
        assert isinstance(x, PyShmemML2DD)
        x_arr = x.get_local_arrow_record_batch().to_pandas(zero_copy_only=True)
        pred = self.model.transform(x_arr)

        dist_pred = PyShmemML1DD(x.N())
        dist_pred.update_from_arrow(pyarrow.array(pred))
        return dist_pred


def pe():
    return c_shmem_my_pe()


def npes():
    return c_shmem_n_pes()

# Initialize the runtime when it is imported
shmem_ml_init()
