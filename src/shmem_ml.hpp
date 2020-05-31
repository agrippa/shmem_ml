#ifndef _SHMEM_ML_HPP
#define _SHMEM_ML_HPP

#include <shmem_ml_utils.hpp>
#include <ShmemMemoryPool.hpp>
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/io/file.h>

template <typename T>
class ShmemML1D {
    public:
        ShmemML1D(int64_t N) {
            _N = N;
            int npes = shmem_n_pes();
            _chunk_size = (_N + npes - 1) / npes;
            _local_slice_start = calculate_local_slice_start(shmem_my_pe());
            _local_slice_end = calculate_local_slice_end(shmem_my_pe());

            shmem_barrier_all();

            pool = ShmemMemoryPool::get();
            std::shared_ptr<arrow::DataType> type = arrow::fixed_size_binary(sizeof(T));

            arrow::FixedSizeBinaryBuilder builder(type, pool);
            CHECK_ARROW(builder.AppendNulls(_chunk_size));
            builder.Finish(&_arr);

            symm_reduce_dest = (T*)shmem_malloc(sizeof(*symm_reduce_dest));
            symm_reduce_src = (T*)shmem_malloc(sizeof(*symm_reduce_src));
            pwork = (T*)shmem_malloc(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(*pwork));
            psync = (long*)shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(*psync));
            assert(symm_reduce_dest && symm_reduce_src && pwork && psync);

            // Ensure all PEs have completed construction
            shmem_barrier_all();
        }

        ShmemML1D(int64_t N, T init_val) : ShmemML1D(N) {
            T* local_slice = raw_slice();
            int64_t local_slice_len = local_slice_end() - local_slice_start();
            for (int64_t i = 0; i < local_slice_len; i++) {
                local_slice[i] = init_val;
            }
            shmem_barrier_all();
        }

        ~ShmemML1D() {
            shmem_free(symm_reduce_dest);
            shmem_free(symm_reduce_src);
            shmem_free(pwork);
            shmem_free(psync);
        }

        inline int64_t N() { return _N; }
        inline int64_t local_slice_start() { return _local_slice_start; }
        inline int64_t local_slice_end() { return _local_slice_end; }
        inline int owning_pe(int64_t global_index) {
            return global_index / _chunk_size;
        }
        inline T* raw_slice() {
            return (T*)_arr->raw_values();
        }

        // Inefficient but simple
        inline T get(int64_t global_index) {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;

            T val;
            shmem_getmem(&val, raw_slice() + offset, sizeof(val), pe);
            return val;
        }

        inline void set(int64_t global_index, T val) {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_putmem(raw_slice() + offset, &val, sizeof(val), pe);
        }

        inline void set_local(int pe, int64_t local_index, T val) {
            shmem_putmem(raw_slice() + local_index, &val, sizeof(val), pe);
        }

        inline void set_local(int64_t local_index, T val) {
            raw_slice()[local_index] = val;
        }

        inline T get_local(int64_t local_index) {
            return raw_slice()[local_index];
        }

        void atomic_add(int64_t global_index, T val);
        T atomic_fetch_add(int64_t global_index, T val);
        T atomic_cas(int64_t global_index, T expected, T update_to);
        T max(T min_val);
        T sum(T zero_val);

        void sync() {
            shmem_barrier_all();
        }

        void save(const char *filename) {
            size_t nbuf = 1024 * 1024;
            void *buf = (void *)malloc(nbuf * sizeof(T));
            assert(buf);

            shmem_barrier_all();
            if (shmem_my_pe() == 0) {
                arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> err =
                    arrow::io::FileOutputStream::Open(std::string(filename));
                std::shared_ptr<arrow::io::FileOutputStream> stream =
                    std::move(err).ValueOrDie();

                // Write the size of this array
                stream->Write(&_N, sizeof(_N));

                for (int pe = 0; pe <  shmem_n_pes(); pe++) {
                    for (int64_t i = calculate_local_slice_start(pe);
                            i < calculate_local_slice_end(pe); i += nbuf) {
                        int64_t to_fetch = calculate_local_slice_end(pe) - i;
                        if (to_fetch > nbuf) to_fetch = nbuf;
                        shmem_getmem(buf,
                                raw_slice() + (i - calculate_local_slice_start(pe)),
                                to_fetch * sizeof(T), pe);
                        CHECK_ARROW(stream->Write(buf, to_fetch * sizeof(T)));
                    }
                }

                CHECK_ARROW(stream->Close());
            }
            shmem_barrier_all();

            free(buf);
        }

        static ShmemML1D<T>* load(const char *filename) {
            shmem_barrier_all();

            arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> err =
                arrow::io::ReadableFile::Open(std::string(filename));
            std::shared_ptr<arrow::io::ReadableFile> stream =
                std::move(err).ValueOrDie();

            int64_t N;
            arrow::Result<int64_t> read = stream->Read(sizeof(N), &N);
            assert(read.ValueOrDie() == sizeof(N));

            ShmemML1D<T>* new_arr = new ShmemML1D<T>(N);

            if (shmem_my_pe() == 0) {
                size_t nbuf = 1024 * 1024;
                void *buf = (void *)malloc(nbuf * sizeof(T));
                assert(buf);

                for (int pe = 0; pe <  shmem_n_pes(); pe++) {
                    for (int64_t i = new_arr->calculate_local_slice_start(pe);
                            i < new_arr->calculate_local_slice_end(pe); i += nbuf) {
                        int64_t to_fetch = new_arr->calculate_local_slice_end(pe) - i;
                        if (to_fetch > nbuf) to_fetch = nbuf;
                        arrow::Result<int64_t> read = stream->Read(to_fetch * sizeof(T), buf);
                        assert(read.ValueOrDie() == to_fetch * sizeof(T));
                        shmem_putmem(new_arr->raw_slice() + (i - new_arr->calculate_local_slice_start(pe)),
                                buf, to_fetch * sizeof(T), pe);
                    }
                }

                free(buf);
            }

            CHECK_ARROW(stream->Close());

            shmem_barrier_all();
            return new_arr;
        }

    private:
        int64_t calculate_local_slice_start(int pe) {
            return pe * _chunk_size;
        }

        int64_t calculate_local_slice_end(int pe) {
            int64_t local_slice_end = calculate_local_slice_start(pe) + _chunk_size;
            if (local_slice_end > _N) {
                local_slice_end = _N;
            }
            return local_slice_end;
        }

        int64_t _N;
        int64_t _chunk_size;
        int64_t _local_slice_start;
        int64_t _local_slice_end;
        ShmemMemoryPool* pool;
        std::shared_ptr<arrow::FixedSizeBinaryArray> _arr;
        T* symm_reduce_dest;
        T* symm_reduce_src;
        T* pwork;
        long *psync;
};

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
int64_t ShmemML1D<int64_t>::atomic_fetch_add(int64_t global_index, int64_t val) {
    int pe = global_index / _chunk_size;
    int64_t offset = global_index % _chunk_size;

    return shmem_int64_atomic_fetch_add(raw_slice() + offset, val, pe);
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

unsigned long long shmem_ml_current_time_us();

#endif
