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

            // Ensure all PEs have completed construction
            shmem_barrier_all();
        }

        ~ShmemML1D() {
        }

        int64_t N() { return _N; }
        int64_t local_slice_start() { return _local_slice_start; }
        int64_t local_slice_end() { return _local_slice_end; }
        T* raw_slice() {
            return (T*)_arr->raw_values();
        }

        // Inefficient but simple
        T get(int64_t global_index) {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;

            T val;
            shmem_getmem(&val, raw_slice() + offset, sizeof(val), pe);
            return val;
        }

        void set(int64_t global_index, T val) {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_putmem(raw_slice() + offset, &val, sizeof(val), pe);
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
};

#endif
