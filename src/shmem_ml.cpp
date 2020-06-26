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
int64_t ShmemML1D<int64_t>::atomic_fetch_add(int64_t global_index, int64_t val) {
    int pe = global_index / _chunk_size;
    int64_t offset = global_index % _chunk_size;

    return shmem_int64_atomic_fetch_add(raw_slice() + offset, val, pe);
}

template<>
void ShmemML1D<int64_t>::atomic_add(int64_t global_index, int64_t val) {
    int pe = global_index / _chunk_size;
    int64_t offset = global_index % _chunk_size;

    shmem_int64_atomic_add(raw_slice() + offset, val, pe);
}

template<>
void ShmemML1D<long long>::atomic_add(int64_t global_index, long long val) {
    int pe = global_index / _chunk_size;
    int64_t offset = global_index % _chunk_size;

    shmem_longlong_atomic_add(raw_slice() + offset, val, pe);
}

template<>
int64_t ShmemML1D<int64_t>::atomic_cas(int64_t global_index, int64_t expected,
        int64_t update_to) {
    int pe = global_index / _chunk_size;
    int64_t offset = global_index % _chunk_size;

    return shmem_int64_atomic_compare_swap(raw_slice() + offset, expected,
            update_to, pe);
}

template<>
long long ShmemML1D<long long>::max(long long min_val) {
    long long my_max = min_val;
    for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
        if (raw_slice()[i] > my_max) {
            my_max = raw_slice()[i];
        }
    }

    *symm_reduce_src = my_max;
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_longlong_max_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
            shmem_n_pes(), pwork, psync);
    shmem_barrier_all();
    return *symm_reduce_dest;
}

template<>
long long ShmemML1D<long long>::sum(long long zero_val) {
    long long my_sum = zero_val;
    for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
        my_sum += raw_slice()[i];
    }

    *symm_reduce_src = my_sum;
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
    shmem_longlong_sum_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
            shmem_n_pes(), pwork, psync);
    shmem_barrier_all();
    return *symm_reduce_dest;
}

template<>
void ShmemML1D<int64_t>::atomic_or(int64_t global_index, int64_t mask) {
    int pe = global_index / _chunk_size;
    int64_t offset = global_index % _chunk_size;

    return shmem_int64_atomic_or(raw_slice() + offset, mask, pe);
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

