#include <time.h>
#include <sys/time.h>

#include "ShmemMemoryPool.hpp"
#include "shmem_ml.hpp"

unsigned long long shmem_ml_current_time_us() {
    struct timespec monotime;
    clock_gettime(CLOCK_MONOTONIC, &monotime);
    return monotime.tv_sec * 1000000ULL + monotime.tv_nsec / 1000;
}

template<>
void ReplicatedShmemML1D<int>::reduce_all_or() {
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_int_or_to_all(this->raw_slice(), this->raw_slice(), _replicated_N, 0, 0,
            shmem_n_pes(), pwork, psync);
    shmem_barrier_all();
}

template<>
void ReplicatedShmemML1D<unsigned>::reduce_all_or() {
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_int_or_to_all((int*)this->raw_slice(), (int*)this->raw_slice(), _replicated_N, 0, 0,
            shmem_n_pes(), (int*)pwork, psync);
    shmem_barrier_all();
}

template<>
void ReplicatedShmemML1D<unsigned>::reduce_all_sum() {
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_int_sum_to_all((int*)this->raw_slice(), (int*)this->raw_slice(),
            _replicated_N, 0, 0, shmem_n_pes(), (int*)pwork, psync);
    shmem_barrier_all();
}

template<>
void ReplicatedShmemML1D<unsigned>::bcast(int src_rank) {
    for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_broadcast32(this->raw_slice(), this->raw_slice(),
            _replicated_N, 0, 0, 0, shmem_n_pes(), psync);
    shmem_barrier_all();
}

template<>
void ReplicatedShmemML1D<double>::reduce_all_sum() {
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_double_sum_to_all((double*)this->raw_slice(),
            (double*)this->raw_slice(), _replicated_N, 0, 0,
            shmem_n_pes(), (double*)pwork, psync);
    shmem_barrier_all();
}

template<>
void ReplicatedShmemML1D<double>::bcast(int src_rank) {
    for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_broadcast64((double*)this->raw_slice(),
            (double*)this->raw_slice(), _replicated_N, 0, 0, 0,
            shmem_n_pes(), psync);
    shmem_barrier_all();
}

void shmem_ml_init() {
    shmem_init();
    arrow::py::import_pyarrow();
}

void shmem_ml_finalize() {
    shmem_finalize();
}

