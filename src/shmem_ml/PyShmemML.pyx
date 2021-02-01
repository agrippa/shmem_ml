# distutils: language = c++
#cython: language_level=3
import pyarrow
from pyarrow.lib cimport *

import sys
import dill
import atexit
import pickle
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.linear_model import SGDRegressor as SK_SGDRegressor

import tensorflow as tf
from tensorflow import keras

from shmem_ml cimport shmem_ml_init as c_shmem_ml_init
from shmem_ml cimport shmem_ml_finalize as c_shmem_ml_finalize
from shmem_ml cimport command_loop as c_command_loop
from shmem_ml cimport shmem_my_pe as c_shmem_my_pe
from shmem_ml cimport shmem_n_pes as c_shmem_n_pes
from shmem_ml cimport end_cmd as c_end_cmd
from shmem_ml cimport is_client_server_mode as c_is_client_server_mode
from shmem_ml cimport send_sgd_fit_cmd as c_send_sgd_fit_cmd
from shmem_ml cimport send_sgd_predict_cmd as c_send_sgd_predict_cmd
from shmem_ml cimport send_sequential_fit_cmd as c_send_sequential_fit_cmd
from shmem_ml cimport send_sequential_predict_cmd as c_send_sequential_predict_cmd

from shmem_ml cimport ShmemML1D, ReplicatedShmemML1D, ShmemML2D, \
        shmem_ml_py_cmd, CMD_DONE, RAND_1D, RAND_2D, APPLY_1D, APPLY_2D, \
        SGD_FIT, SGD_PREDICT, SEQUENTIAL_FIT, SEQUENTIAL_PREDICT


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
    cdef public (int64_t,) shape

    def __cinit__(self, int64_t N):
        if N == 0:
            self.c_vec = NULL
        else:
            self.c_vec = new ShmemML1D[double](N)
        self.shape = (N,)

    def __dealloc__(self):
        # The challenge with automatically de-allocating the underlying C data
        # structure is that there may be no references to this in Python, but we
        # may still be able to access it (e.g. in client-server mode where we
        # access by ID). Need to consider a better solution than just not
        # freeing memory. Maybe on servers we manually place a reference on all
        # of these objects and don't release until receiving an explicit destroy
        # command.
        pass
        # del self.c_vec

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

    def gather(self):
        # Very inefficient (for now)
        gathered = np.zeros(self.N())
        for i in range(self.N()):
            gathered[i] = self.get(i)
        return gathered

    def apply(self, f):
        serialized = dill.dumps(f)
        self.c_vec.send_apply_1d_cmd(serialized, len(serialized))

        arrow_arr = pyarrow_wrap_array(self.c_vec.get_arrow_array())
        np_arr = arrow_arr.to_numpy(zero_copy_only=True, writable=False)
        out_np_arr = np.zeros(len(np_arr))
        for i in range(len(np_arr)):
            out_np_arr[i] = f(self.local_slice_start() + i, np_arr[i], self)

        new_dist_arr = PyShmemML1DD(self.N())
        new_dist_arr.update_from_arrow(pyarrow.array(out_np_arr))
        new_dist_arr.sync()
        c_end_cmd()
        return new_dist_arr

    def send_rand_1d_cmd(self):
        self.c_vec.send_rand_1d_cmd()

    def get_id(self):
        return self.c_vec.get_id()

    def mse(self, other):
        assert isinstance(other, PyShmemML1DD) or \
                (isinstance(other, PyShmemML2DD) and other.shape[1] == 1), str(type(other), other.shape)
        local_self = self.get_local_arrow_array().to_numpy(zero_copy_only=True,
                writable=False)
        if isinstance(other, PyShmemML1DD):
            local_other = other.get_local_arrow_array().to_numpy(
                    zero_copy_only=True, writable=False)
        else:
            local_other = other.get_local_arrow_table().column(0).to_numpy()

        local_mse = tf.compat.v1.losses.mean_squared_error(local_self,
                local_other)

        global_mse = PyReplicatedShmemML1DD(1)
        np_arr = np.zeros(1)
        np_arr[0] = local_mse
        global_mse.update_from_arrow(pyarrow.array(np_arr))
        global_mse.reduce_all_sum()
        return global_mse.get(0).as_py() / float(npes())


cdef class PyShmemML2DD:
    cdef ShmemML2D* c_mat
    cdef public (int64_t, int64_t) shape

    def __cinit__(self, int64_t M, int64_t N):
        if M == 0 and N == 0:
            self.c_mat = NULL
        else:
            self.c_mat = new ShmemML2D(M, N)
        self.shape = (M, N)

    def __dealloc__(self):
        pass
        # del self.c_mat

    def N(self):
        return self.c_mat.N()

    def M(self):
        return self.c_mat.M()

    def sync(self):
        self.c_mat.sync()

    def rows_per_pe(self):
        return self.c_mat.rows_per_pe()

    def get(self, int64_t row, int64_t col):
        return self.c_mat.get(row, col)

    def get_local_arrow_table(self):
        return pyarrow_wrap_table(self.c_mat.get_arrow_table())

    def update_from_arrow(self, arrow_table):
        cdef shared_ptr[CTable] tab = pyarrow_unwrap_table(arrow_table)
        self.c_mat.update_from_arrow_table(tab)

    def gather(self):
        # Very inefficient (for now)
        gathered = np.zeros((self.M(), self.N()))
        for i in range(self.M()):
            for j in range(self.N()):
                gathered[i, j] = self.get(i, j)
        return gathered

    def apply(self, f):
        serialized = dill.dumps(f)
        self.c_mat.send_apply_2d_cmd(serialized, len(serialized))

        arrow_table = self.get_local_arrow_table()
        arrow_table = arrow_table.to_pandas(zero_copy_only=True, split_blocks=True)
        result = pd.DataFrame(np.zeros((arrow_table.shape)))

        for i in range(arrow_table.shape[0]):
            for j in range(arrow_table.shape[1]):
                result.iloc[i, j] = f(pe() * self.rows_per_pe() + i, j, arrow_table.iloc[i, j], self)
        dist_mat = PyShmemML2DD(self.M(), self.N())
        dist_mat.update_from_arrow(pyarrow.table(result))
        dist_mat.sync()
        c_end_cmd()
        return dist_mat

    def send_rand_2d_cmd(self):
        self.c_mat.send_rand_2d_cmd()

    def get_id(self):
        return self.c_mat.get_id()


cdef class PyReplicatedShmemML1DD:
    cdef ReplicatedShmemML1D[double]* c_vec

    def __cinit__(self, int64_t N):
        self.c_vec = new ReplicatedShmemML1D[double](N)

    def __dealloc__(self):
        pass
        # del self.c_vec

    def update_from_arrow(self, arrow_arr):
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(arrow_arr)
        self.c_vec.update_from_arrow_array(arr)

    def reduce_all_sum(self):
        return self.c_vec.reduce_all_sum()

    def get_local_arrow_array(self):
        return pyarrow_wrap_array(self.c_vec.get_arrow_array())

    def bcast(self, src_rank):
        self.c_vec.bcast(src_rank)

    def get(self, local_index):
        return self.get_local_arrow_array()[local_index]


def shmem_ml_command_loop():
    cdef ShmemML1D[double]* c_vec
    cdef ShmemML2D* c_mat
    cdef shmem_ml_py_cmd py_cmd
    cdef char* s
    cdef bytes py_s
    done = False

    while not done:
        py_cmd = c_command_loop()
        if <int>py_cmd.get_cmd() == <int>CMD_DONE:
            done = True
        elif <int>py_cmd.get_cmd() == <int>RAND_1D:
            c_vec = <ShmemML1D[double]*>py_cmd.get_arr()
            wrapped_vec = PyShmemML1DD(0)
            wrapped_vec.c_vec = c_vec
            rand(wrapped_vec)
        elif <int>py_cmd.get_cmd() == <int>RAND_2D:
            c_mat = <ShmemML2D*>py_cmd.get_arr()
            wrapped_mat = PyShmemML2DD(0, 0)
            wrapped_mat.c_mat = c_mat
            rand(wrapped_mat)
        elif <int>py_cmd.get_cmd() == <int>APPLY_1D:
            c_vec = <ShmemML1D[double]*>py_cmd.get_arr()
            wrapped_vec = PyShmemML1DD(0)
            wrapped_vec.c_vec = c_vec

            s = py_cmd.get_str()
            l = py_cmd.get_str_length()
            py_s = s[:l]
            f = dill.loads(py_s)

            wrapped_vec.apply(f)
        elif <int>py_cmd.get_cmd() == <int>APPLY_2D:
            c_mat = <ShmemML2D*>py_cmd.get_arr()
            wrapped_mat = PyShmemML2DD(0, 0)
            wrapped_mat.c_mat = c_mat

            s = py_cmd.get_str()
            l = py_cmd.get_str_length()
            py_s = s[:l]
            f = dill.loads(py_s)

            wrapped_mat.apply(f)
        elif <int>py_cmd.get_cmd() == <int>SGD_FIT:
            c_mat = <ShmemML2D*>py_cmd.get_arr()
            c_vec = <ShmemML1D[double]*>py_cmd.get_arr_2()

            wrapped_mat = PyShmemML2DD(0, 0)
            wrapped_mat.c_mat = c_mat

            wrapped_vec = PyShmemML1DD(0)
            wrapped_vec.c_vec = c_vec

            s = py_cmd.get_str()
            l = py_cmd.get_str_length()
            py_s = s[:l]
            m = dill.loads(py_s)

            m.fit(wrapped_mat, wrapped_vec)
        elif <int>py_cmd.get_cmd() == <int>SGD_PREDICT:
            c_mat = <ShmemML2D*>py_cmd.get_arr()

            wrapped_mat = PyShmemML2DD(0, 0)
            wrapped_mat.c_mat = c_mat

            s = py_cmd.get_str()
            l = py_cmd.get_str_length()
            py_s = s[:l]
            m = dill.loads(py_s)

            m.predict(wrapped_mat)
        elif <int>py_cmd.get_cmd() == <int>SEQUENTIAL_FIT:
            c_mat = <ShmemML2D*>py_cmd.get_arr()
            c_vec = <ShmemML1D[double]*>py_cmd.get_arr_2()

            wrapped_mat = PyShmemML2DD(0, 0)
            wrapped_mat.c_mat = c_mat

            wrapped_vec = PyShmemML1DD(0)
            wrapped_vec.c_vec = c_vec

            s = py_cmd.get_str()
            l = py_cmd.get_str_length()
            py_s = s[:l]

            fit_config = dill.loads(py_s)
            assert 'config' in fit_config and 'weights' in fit_config and 'compile_config' in fit_config
            assert fit_config['compile_config'] is not None
            new_model = keras.Sequential.from_config(fit_config['config'])
            new_model.set_weights(fit_config['weights'])
            new_model.compile(**fit_config['compile_config'])
            del fit_config['config']
            del fit_config['weights']
            del fit_config['compile_config']

            m = Sequential()
            m.model = new_model
            m.fit(wrapped_mat, wrapped_vec, **fit_config)
        elif <int>py_cmd.get_cmd() == <int>SEQUENTIAL_PREDICT:
            c_mat = <ShmemML2D*>py_cmd.get_arr()

            wrapped_mat = PyShmemML2DD(0, 0)
            wrapped_mat.c_mat = c_mat

            s = py_cmd.get_str()
            l = py_cmd.get_str_length()
            py_s = s[:l]

            fit_config = dill.loads(py_s)
            assert 'config' in fit_config and 'weights' in fit_config and 'compile_config' in fit_config
            assert fit_config['compile_config'] is not None
            new_model = keras.Sequential.from_config(fit_config['config'])
            new_model.set_weights(fit_config['weights'])
            new_model.compile(**fit_config['compile_config'])
            del fit_config['config']
            del fit_config['weights']
            del fit_config['compile_config']

            m = Sequential()
            m.model = new_model
            m.predict(wrapped_mat, **fit_config)
        else:
            raise Exception('Unexpected command ' + str(<int>py_cmd.get_cmd()))


def rand(vec):
    if isinstance(vec, PyShmemML1DD):
        vec.send_rand_1d_cmd()

        arr = vec.get_local_arrow_array()
        np_arr = arr.to_numpy(zero_copy_only=False, writable=True)
        np_arr[:] = np.random.rand(np_arr.shape[0])
        pyarr = pyarrow.array(np_arr)
        vec.update_from_arrow(pyarr)
        c_end_cmd()
        return vec
    elif isinstance(vec, PyShmemML2DD):
        vec.send_rand_2d_cmd()

        np_arr = np.random.rand(vec.rows_per_pe(), vec.N())
        pd_arr = pd.DataFrame(np_arr)
        arrow_table = pyarrow.Table.from_pandas(pd_arr)
        vec.update_from_arrow(arrow_table)
        c_end_cmd()
        return vec
    else:
        assert False, str(type(vec))

def _training_driver(model, x_arr, y_arr, epochs, **custom_args):
    dist_weights_grad = None

    for it in range(epochs):

        model._fit_one_epoch(x_arr, y_arr, **custom_args)

        after_weights = model._copy_weights()
        arrow_weights = pyarrow.array(after_weights)
        if dist_weights_grad is None:
            dist_weights_grad = PyReplicatedShmemML1DD(len(arrow_weights))
        dist_weights_grad.update_from_arrow(arrow_weights)
        dist_weights_grad.reduce_all_sum()

        all_weights_grads = dist_weights_grad.get_local_arrow_array().to_numpy(zero_copy_only=True)
        all_weights_grads = all_weights_grads / npes()
        model._update_weights(all_weights_grads)


class SGDRegressor:
    def __init__(self, max_iter=1, random_state=32):
        self.model = SK_SGDRegressor(max_iter=1, tol=None,
                random_state=random_state, learning_rate='constant')
        self.max_iter = max_iter

    def _copy_weights(self):
        save_weights = np.zeros(len(self.model.coef_) + len(self.model.intercept_))
        save_weights[0:len(self.model.coef_)] = self.model.coef_
        save_weights[len(self.model.coef_):] = self.model.intercept_
        return save_weights

    def _compute_grad(self, prev_weights, after_weights):
        return after_weights - prev_weights

    def _update_weights(self, new_weights):
        self.model.coef_ = new_weights[0:len(self.model.coef_)]
        self.model.intercept_ = new_weights[len(self.model.coef_):]

    def _fit_one_epoch(self, x_arr, y_arr):
        self.model.partial_fit(x_arr, y_arr)

    def fit(self, x, y):
        assert isinstance(x, PyShmemML2DD)
        assert isinstance(y, PyShmemML1DD)

        if is_client_server_mode():
            serialized = dill.dumps(self)
            serialized_len = len(serialized)
            c_send_sgd_fit_cmd(x.get_id(), y.get_id(), serialized, serialized_len)

        x_arr = x.get_local_arrow_table()
        x_arr = x_arr.to_pandas(zero_copy_only=True, split_blocks=True)

        y_arr = y.get_local_arrow_array()
        y_arr = y_arr.to_numpy(zero_copy_only=True)

        _training_driver(self, x_arr, y_arr, self.max_iter)

        if is_client_server_mode():
            c_end_cmd()


    def predict(self, x):
        assert isinstance(x, PyShmemML2DD)

        serialized = dill.dumps(self)
        c_send_sgd_predict_cmd(x.get_id(), serialized, len(serialized))

        x_arr = x.get_local_arrow_table().to_pandas(zero_copy_only=True, split_blocks=True)

        pred = self.model.predict(x_arr)

        dist_pred = PyShmemML1DD(x.M())
        dist_pred.update_from_arrow(pyarrow.array(pred))
        c_end_cmd()
        return dist_pred


class Sequential:
    def __init__(self, layers=None, name=None):
        self.model = keras.Sequential(layers=layers, name=name)
        self.compile_config = None

    def _copy_weights(self):
        nweights = 0
        for layer in self.model.layers:
            for weights in layer.get_weights():
                nweights += weights.size

        save_weights = np.zeros(nweights)
        nweights = 0
        for layer in self.model.layers:
            for weights in layer.get_weights():
                save_weights[nweights:nweights + weights.size] = weights.flatten()
                nweights += weights.size

        return save_weights


    def _compute_grad(self, prev_weights, after_weights):
        return after_weights - prev_weights


    def _update_weights(self, updated_weights):
        nweights = 0
        for layer in self.model.layers:
            packed_weights = []
            for weights in layer.get_weights():
                new_weights = updated_weights[nweights:nweights + weights.size]
                packed_weights.append(new_weights.reshape(weights.shape))
                nweights += weights.size
            layer.set_weights(tuple(packed_weights))


    def _fit_one_epoch(self, x_arr, y_arr, batch_size=32):
        for batch_offset in range(0, x_arr.shape[0], batch_size):
            self.model.train_on_batch(x_arr.values[batch_offset:batch_offset + batch_size],
                    y_arr[batch_offset:batch_offset + batch_size])


    def fit(self, x=None, y=None, batch_size=32, epochs=1):
        assert isinstance(x, PyShmemML2DD)
        assert isinstance(y, PyShmemML1DD)

        if is_client_server_mode():
            serialized = {'config': self.model.get_config(),
                          'weights': self.model.get_weights(),
                          'batch_size': batch_size,
                          'epochs': epochs,
                          'compile_config': self.compile_config}
            serialized = dill.dumps(serialized)
            serialized_len = len(serialized)
            c_send_sequential_fit_cmd(x.get_id(), y.get_id(), serialized, serialized_len)

        x_arr = x.get_local_arrow_table().to_pandas(zero_copy_only=True, split_blocks=True)
        y_arr = y.get_local_arrow_array().to_numpy(zero_copy_only=True)

        _training_driver(self, x_arr, y_arr, epochs, batch_size=batch_size)

        if is_client_server_mode():
            c_end_cmd()


    def predict(self, x):
        assert isinstance(x, PyShmemML2DD)

        if is_client_server_mode():
            # Optimization, so we aren't serializing if we don't have to
            serialized = {'config': self.model.get_config(),
                          'weights': self.model.get_weights(),
                          'compile_config': self.compile_config}
            serialized = dill.dumps(serialized)
            serialized_len = len(serialized)
            c_send_sequential_predict_cmd(x.get_id(), serialized, len(serialized))

        x_arr = x.get_local_arrow_table().to_pandas(zero_copy_only=True, split_blocks=True)

        pred = self.model.predict(x_arr.values)

        dist_pred = PyShmemML2DD(x.M(), pred.shape[1])
        dist_pred.update_from_arrow(pyarrow.table(pd.DataFrame(pred)))

        if is_client_server_mode():
            c_end_cmd()

        return dist_pred

    def compile(self, optimizer='rmsprop', loss=None):
        self.compile_config = {'optimizer': optimizer,
                               'loss': loss}
        self.model.compile(optimizer=optimizer, loss=loss)

    def evaluate(x=None, y=None):
        raise Exception('unsupported')

    def train_on_batch(x, y=None):
        raise Exception('unsupported')

    def __getattr__(self, name):
        if callable(getattr(keras.Sequential, name)):
            def method(*args, **kwargs):
                class_method = getattr(keras.Sequential, name)
                return class_method(self.model, *args, **kwargs)
            return method
        else:
            return getattr(self.model, name)

def pe():
    return c_shmem_my_pe()

def npes():
    return c_shmem_n_pes()

def is_client_server_mode():
    return c_is_client_server_mode()

# Initialize the runtime when it is imported
shmem_ml_init()
