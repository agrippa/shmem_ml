#ifndef _SHMEM_ML_HPP
#define _SHMEM_ML_HPP

#include <shmem_ml_utils.hpp>
#include <mailbox.hpp>
#include <mailbox_buffer.hpp>
#include <ShmemMemoryPool.hpp>
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/io/file.h>
#include <set>

#define ATOMICS_AS_MSGS

#define SHMEMML_MAX(_a, _b) (((_a) > (_b)) ? (_a) : (_b))

#ifdef ATOMICS_AS_MSGS
#define MAX_BUFFERED_ATOMICS 1024

template <typename T>
class ShmemML1D;

typedef enum {
    CAS = 0,
    DONE
} atomics_msg_op_t;

template <typename T>
struct atomics_msg_t {
    int64_t local_index;
    T cmp;
    T val;
    atomics_msg_op_t op;
};

template<typename T>
using atomics_msg_result_handler = void (*)(ShmemML1D<T>* arr,
        int64_t global_index, atomics_msg_op_t, T prev_val, T new_val);

#endif

class ShmemML1DIndex {
    private:
        std::set<int64_t> indices;

    public:
        ShmemML1DIndex() { }

        void add(int64_t i) {
            indices.insert(i);
        }

        void clear() {
            indices.clear();
        }

        std::set<int64_t>::iterator begin() { return indices.begin(); }
        std::set<int64_t>::iterator end() { return indices.end(); }
};

template <typename T>
class ShmemML1D {
    public:
        ShmemML1D(int64_t N,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_result_handler<T> _atomics_cb = NULL
#endif
                ) {
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

            pwork = (T*)shmem_malloc(SHMEMML_MAX(max_shmem_reduction_n / 2 + 1,
                        SHMEM_REDUCE_MIN_WRKDATA_SIZE) * sizeof(*pwork));
            psync = (long*)shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(*psync));
            assert(symm_reduce_dest && symm_reduce_src && pwork && psync);

#ifdef ATOMICS_AS_MSGS
            mailbox_init(&atomics_mailbox, 32 * 1024 * 1024);
            mailbox_buffer_init(&atomics_mailbox_buffer, &atomics_mailbox,
                    npes, sizeof(atomics_msg_t<T>), MAX_BUFFERED_ATOMICS);
            buffered_atomics = (atomics_msg_t<T>*)malloc(
                    MAX_BUFFERED_ATOMICS * sizeof(*buffered_atomics));
            assert(buffered_atomics);
            n_done_pes = 0;
            atomics_cb = _atomics_cb;
#endif

            // Ensure all PEs have completed construction
            shmem_barrier_all();
        }

        void clear(T init_val) {
            T* local_slice = this->raw_slice();
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

        template <typename lambda>
        inline void apply_ip(lambda&& l) {
            const int64_t slice_size = _local_slice_end - _local_slice_start;
            T* slice = raw_slice();
            for (int64_t i = 0; i < slice_size; i++) {
                l(_local_slice_start + i, i, slice[i]);
            }
        }

        template <typename lambda>
        inline void apply_ip(lambda& l) {
            const int64_t slice_size = _local_slice_end - _local_slice_start;
            T* slice = raw_slice();
            for (int64_t i = 0; i < slice_size; i++) {
                l(_local_slice_start + i, i, slice[i]);
            }
        }

        template <typename lambda>
        inline void apply_ip(ShmemML1DIndex& global_indices, lambda&& l) {
            T* slice = raw_slice();
            for (auto i = global_indices.begin(), e = global_indices.end();
                    i != e; i++) {
                int64_t global = *i;
                int64_t local = global - _local_slice_start;
                l(global, local, slice[local]);
            }
        }

        template <typename lambda>
        inline void apply_ip(ShmemML1DIndex* global_indices, lambda&& l) {
            T* slice = raw_slice();
            for (auto i = global_indices->begin(), e = global_indices->end();
                    i != e; i++) {
                int64_t global = *i;
                int64_t local = global - _local_slice_start;
                l(global, local, slice[local]);
            }
        }

        void atomic_add(int64_t global_index, T val);
        T atomic_fetch_add(int64_t global_index, T val);
        T atomic_cas(int64_t global_index, T expected, T update_to);
        void atomic_or(int64_t global_index, T mask);
        T max(T min_val);
        T sum(T zero_val);

#ifdef ATOMICS_AS_MSGS
        void atomic_cas_msg(int64_t global_index, T expected, T update_to) {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;

            atomics_msg_t<T> msg;
            msg.local_index = offset;
            msg.cmp = expected;
            msg.val = update_to;
            msg.op = CAS;

            int success;
            do {
                success = mailbox_buffer_send(&msg, sizeof(msg),
                        pe, 100, &atomics_mailbox_buffer);
                if (!success) {
                    process_atomic_msgs();
                }
            } while (!success);
        }
#endif

        void sync() {
#ifdef ATOMICS_AS_MSGS
            atomics_msg_t<T> msg;
            msg.op = DONE;
            for (int p = 0; p < shmem_n_pes(); p++) {
                int success;
                do {
                    success = mailbox_buffer_send(&msg, sizeof(msg), p, 100,
                            &atomics_mailbox_buffer);
                    if (!success) {
                        process_atomic_msgs();
                    }
                } while (!success);
            }

            int success;
            do {
                success = mailbox_buffer_flush(&atomics_mailbox_buffer, 100);
                if (!success) {
                    process_atomic_msgs();
                }
            } while (!success);

            while (n_done_pes < shmem_n_pes()) {
                process_atomic_msgs();
            }

            n_done_pes = 0;
#endif
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

    protected:
        T* pwork;
        long *psync;

    private:
        void process_atomic_msgs() {
            int success;
            do {
                T old_val, new_val;
                size_t msg_len;
                success = mailbox_recv(buffered_atomics,
                        MAX_BUFFERED_ATOMICS * sizeof(*buffered_atomics),
                        &msg_len, &atomics_mailbox);
                if (success) {
                    assert(msg_len % sizeof(*buffered_atomics) == 0);
                    size_t nmsgs = msg_len / sizeof(*buffered_atomics);
                    for (int m = 0; m < nmsgs; m++) {
                        atomics_msg_t<T> *msg = &buffered_atomics[m];
                        switch (msg->op) {
                            case (CAS):
                                old_val = new_val = raw_slice()[msg->local_index];
                                if (old_val == msg->cmp) {
                                    raw_slice()[msg->local_index] = new_val = msg->val;
                                }
                                break;
                            case (DONE):
                                n_done_pes++;
                                break;
                            default:
                                abort();
                        }
                        if (atomics_cb) {
                            atomics_cb(this,
                                    _local_slice_start + msg->local_index,
                                    msg->op, old_val, new_val);
                        }
                    }
                }
            } while (success);
        }

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

#ifdef ATOMICS_AS_MSGS
        mailbox_t atomics_mailbox;
        mailbox_buffer_t atomics_mailbox_buffer;
        atomics_msg_t<T> *buffered_atomics;
        int n_done_pes;
        atomics_msg_result_handler<T> atomics_cb;
#endif
};

template<>
void ShmemML1D<int64_t>::atomic_add(int64_t global_index, int64_t val);

template<>
void ShmemML1D<long long>::atomic_add(int64_t global_index, long long val);

template<>
int64_t ShmemML1D<int64_t>::atomic_fetch_add(int64_t global_index, int64_t val);

template<>
int64_t ShmemML1D<int64_t>::atomic_cas(int64_t global_index, int64_t expected,
        int64_t update_to);

template<>
long long ShmemML1D<long long>::max(long long min_val);

template<>
void ShmemML1D<int64_t>::atomic_or(int64_t global_index, int64_t mask);

template<>
long long ShmemML1D<long long>::sum(long long zero_val);

template <typename T>
class ReplicatedShmemML1D : public ShmemML1D<T> {
    private:
        int64_t _replicated_N;

    public:
        ReplicatedShmemML1D(int64_t N) : ShmemML1D<T>(N * shmem_n_pes(), (unsigned)N) {
            _replicated_N = N;
        }

        inline int64_t N() { return _replicated_N; }
        inline int64_t local_slice_start() { return 0; }
        inline int64_t local_slice_end() { return _replicated_N; }
        inline int owning_pe(int64_t global_index) {
            return shmem_my_pe();
        }

        // Inefficient but simple
        inline T get(int64_t global_index) {
            return this->raw_slice()[global_index];
        }

        inline void set(int64_t global_index, T val) {
            this->raw_slice()[global_index] = val;
        }

        inline void set_local(int pe, int64_t local_index, T val) {
            throw std::runtime_error("set_local unsupported on ReplicatedShmemML1D");
        }

        inline void set_local(int64_t local_index, T val) {
            throw std::runtime_error("set_local unsupported on ReplicatedShmemML1D");
        }

        inline T get_local(int64_t local_index) {
            throw std::runtime_error("get_local unsupported on ReplicatedShmemML1D");
        }

        template <typename lambda>
        inline void apply_ip(lambda&& l) {
            T* slice = this->raw_slice();
            for (int64_t i = 0; i < _replicated_N; i++) {
                l(i, i, slice[i]);
            }
        }

        template <typename lambda>
        inline void apply_ip(lambda& l) {
            T* slice = this->raw_slice();
            for (int64_t i = 0; i < _replicated_N; i++) {
                l(i, i, slice[i]);
            }
        }

        void atomic_add(int64_t global_index, T val) {
            this->raw_slice()[global_index] += val;
        }

        T atomic_fetch_add(int64_t global_index, T val) {
            T old = this->raw_slice()[global_index];
            this->raw_slice()[global_index] = old + val;
            return old;
        }

        T atomic_cas(int64_t global_index, T expected, T update_to) {
            T old = this->raw_slice()[global_index];
            if (old == expected) {
                this->raw_slice()[global_index] = update_to;
            }
            return old;
        }

        void atomic_or(int64_t global_index, T mask) {
            this->raw_slice()[global_index] |= mask;
        }

        T sum(T zero_val) {
            T s = zero_val;
            T* slice = this->raw_slice();
            for (int64_t i = 0; i < _replicated_N; i++) {
                s += slice[i];
            }
            return s;
        }

        T max(T min_val) {
            T s = min_val;
            T* slice = this->raw_slice();
            for (int64_t i = 0; i < _replicated_N; i++) {
                if (slice[i] > s) {
                    s = slice[i];
                }
            }
            return s;
        }

        void save(const char *filename) {
            throw std::runtime_error("save unsupported on ReplicatedShmemML1D");
        }

        static ShmemML1D<T>* load(const char *filename) {
            throw std::runtime_error("load unsupported on ReplicatedShmemML1D");
        }

        void reduce_all_or();

        void zero() {
            memset(this->raw_slice(), 0x00, _replicated_N * sizeof(T));
            shmem_barrier_all();
        }
};

template<>
void ReplicatedShmemML1D<int>::reduce_all_or();

template<>
void ReplicatedShmemML1D<unsigned>::reduce_all_or();

unsigned long long shmem_ml_current_time_us();

#endif
