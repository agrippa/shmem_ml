#ifndef _SHMEM_ML_HPP
#define _SHMEM_ML_HPP

#include <shmem_ml_utils.hpp>
#include <shmem_ml_cmd.hpp>
#include <mailbox.hpp>
#include <mailbox_buffer.hpp>
#include <ShmemMemoryPool.hpp>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/io/file.h>
#include <arrow/python/pyarrow.h>

#include <set>
#include <sstream>
#include <dlfcn.h>

// #define VERBOSE_CMD
#define UNUSED_VAR(x) (void)(x)
#define BITS_PER_BYTE 8

#define ATOMICS_AS_MSGS

#define SHMEMML_MAX(_a, _b) (((_a) > (_b)) ? (_a) : (_b))

extern volatile int this_pe_has_exited;
extern int id_counter;
extern mailbox_t cmd_mailbox;
extern shmem_ctx_t cmd_ctx;

void *aborting_thread(void *user_data);

bool setup_client_server();
void add_array_to_namespace(unsigned id, void *arr);
void* lookup_array_in_namespace(unsigned id);
void* delete_array_in_namespace(unsigned id);

void send_sgd_fit_cmd(unsigned x_id, unsigned y_id, char *serialized_model,
        int serialized_model_length);

void send_sgd_predict_cmd(unsigned x_id, char *serialized_model,
        int serialized_model_length);

void send_sequential_fit_cmd(unsigned x_id, unsigned y_id, char *serialized_model,
        int serialized_model_length);

void send_sequential_predict_cmd(unsigned x_id, char *serialized_model,
        int serialized_model_length);

template <typename T>
class ShmemML1D;

class ShmemML2D;

class shmem_ml_py_cmd {
    private:
        shmem_ml_command cmd;
        void *arr;
        char *str;
        int str_length;
        void *arr_2;

    public:
        shmem_ml_py_cmd() : cmd(CMD_INVALID), arr(NULL), str(NULL),
                str_length(0), arr_2(NULL) { }
        shmem_ml_py_cmd(shmem_ml_command _cmd, void *_arr, char *_str,
                int _str_length, void *_arr_2) : cmd(_cmd), arr(_arr),
            str(_str), str_length(_str_length), arr_2(_arr_2) { }

        shmem_ml_command get_cmd() { return cmd; }
        void* get_arr() { return arr; }
        char* get_str() { return str; }
        int get_str_length() { return str_length; }
        void *get_arr_2() { return arr_2; }
};

shmem_ml_py_cmd command_loop();

#ifdef ATOMICS_AS_MSGS

#define MAX_BUFFERED_ATOMICS 4096

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
using atomics_msg_1d_result_handler = void (*)(ShmemML1D<T>* arr,
        int64_t global_index, atomics_msg_op_t, T prev_val, T new_val);

template<typename T>
using atomics_msg_2d_result_handler = void (*)(ShmemML2D* arr,
        int64_t row, int64_t col, atomics_msg_op_t, T prev_val, T new_val);
#endif

#define _BITS_PER_BYTE 8
#define _BITS_PER_INT (sizeof(unsigned) * _BITS_PER_BYTE)

class ShmemML1DIndex {
    private:
        int64_t N;
        unsigned *bitvector;
        unsigned bitvector_size;

        int64_t *list;
        int64_t list_len;

        static int index_compare(const void *a, const void *b) {
            const int64_t *ia = (const int64_t *)a;
            const int64_t *ib = (const int64_t *)b;
            return *ia - *ib;
        }

    public:
        ShmemML1DIndex(int64_t _N) : N(_N), list_len(0) {
            bitvector_size = ((N + _BITS_PER_INT - 1) /
                    _BITS_PER_INT);
            bitvector = (unsigned *)malloc(bitvector_size * sizeof(*bitvector));
            list = (int64_t *)malloc(N * sizeof(*list));
            assert(bitvector && list);
            memset(bitvector, 0x00, bitvector_size * sizeof(*bitvector));
        }

        void add(int64_t i) {
            const unsigned word_index = i / _BITS_PER_INT;
            const int bit_index = i - (word_index * _BITS_PER_INT);
            const unsigned mask = ((unsigned)1 << bit_index);
            if ((bitvector[word_index] & mask) == 0) {
                bitvector[word_index] = (bitvector[word_index] | mask);
                list[list_len++] = i;
            }
        }

        void clear() {
            list_len = 0;
            memset(bitvector, 0x00, bitvector_size * sizeof(*bitvector));
        }

        int64_t *begin() {
            qsort(list, list_len, sizeof(*list), index_compare);
            return list;
        }
        int64_t *end() {
            return list + list_len;
        }
};

template <typename T>
class ShmemML1D_Base {
    public:
        ShmemML1D_Base(int64_t N, shmem_ml_cmd<T>&& create_cmd,
                shmem_ml_cmd<T>&& _destroy_cmd,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_1d_result_handler<T> _atomics_cb = NULL
#endif
                ) : destroy_cmd(_destroy_cmd) {
            assert(sizeof(int64_t) == sizeof(double));
            id = id_counter++;
            _N = N;
            int npes = shmem_n_pes();
            _chunk_size = (_N + npes - 1) / npes;
            _local_slice_start = calculate_local_slice_start(shmem_my_pe());
            _local_slice_end = calculate_local_slice_end(shmem_my_pe());

            send_cmd(&create_cmd);
            shmem_barrier_all();

            pool = ShmemMemoryPool::get();

            symm_reduce_dest = (T*)shmem_malloc(sizeof(*symm_reduce_dest));
            symm_reduce_src = (T*)shmem_malloc(sizeof(*symm_reduce_src));

            pwork = (T*)shmem_malloc(SHMEMML_MAX(max_shmem_reduction_n / 2 + 1,
                        SHMEM_REDUCE_MIN_WRKDATA_SIZE) * sizeof(*pwork));
            psync = (long*)shmem_malloc(SHMEMML_MAX(SHMEM_REDUCE_SYNC_SIZE, SHMEM_BCAST_SYNC_SIZE) * sizeof(*psync));
            assert(symm_reduce_dest && symm_reduce_src && pwork && psync);

#ifdef ATOMICS_AS_MSGS
            int max_mailbox_buffers = -1;
            if (getenv("SHMEM_ML_MAX_MAILBOX_BUFFERS")) {
                max_mailbox_buffers = atoi(getenv("SHMEM_ML_MAX_MAILBOX_BUFFERS"));
            }
            int n_mailbox_buffers = npes;
            if (n_mailbox_buffers < 256) n_mailbox_buffers = 256;
            if (max_mailbox_buffers != -1 && n_mailbox_buffers > max_mailbox_buffers) {
                n_mailbox_buffers = max_mailbox_buffers;
            }
            mailbox_init(&atomics_mailbox, 32 * 1024 * 1024);
            mailbox_buffer_init(&atomics_mailbox_buffer, &atomics_mailbox,
                    npes, sizeof(atomics_msg_t<T>), MAX_BUFFERED_ATOMICS,
                    n_mailbox_buffers);
            buffered_atomics = (atomics_msg_t<T>*)malloc(
                    MAX_BUFFERED_ATOMICS * sizeof(*buffered_atomics));
            assert(buffered_atomics);
            n_done_pes = 0;
            atomics_cb = _atomics_cb;

            // Ensure all PEs have completed construction
            shmem_barrier_all();
#endif
            add_array_to_namespace(id, this);
            end_cmd();
        }

        void clear(T init_val) {
            shmem_ml_clear_1d<T> cmd(id, init_val);
            send_cmd(&cmd);

            T* local_slice = this->raw_slice();

            int64_t local_slice_len = local_slice_end() - local_slice_start();
            for (int64_t i = 0; i < local_slice_len; i++) {
                local_slice[i] = init_val;
            }

            shmem_barrier_all();
            end_cmd();
        }

        ~ShmemML1D_Base() {
            send_cmd(&destroy_cmd);

            // Collective destruction
            shmem_barrier_all();

#ifdef ATOMICS_AS_MSGS
            mailbox_destroy(&atomics_mailbox);
#endif
            shmem_free(symm_reduce_dest);
            shmem_free(symm_reduce_src);
            shmem_free(pwork);
            shmem_free(psync);
            end_cmd();
        }

        inline int64_t N() { return _N; }
        inline int64_t local_slice_start() { return _local_slice_start; }
        inline int64_t local_slice_end() { return _local_slice_end; }
        inline int owning_pe(int64_t global_index) {
            return global_index / _chunk_size;
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

        virtual void atomic_add(int64_t global_index, T val) = 0;
        virtual T atomic_fetch_add(int64_t global_index, T val) = 0;
        virtual T atomic_cas(int64_t global_index, T expected, T update_to) = 0;
        virtual void atomic_or(int64_t global_index, T mask) = 0;
        virtual T max(T min_val) = 0;
        virtual T sum(T zero_val) = 0;

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
            shmem_ml_sync_1d<T> cmd(id);
            send_cmd(&cmd);

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
            end_cmd();
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

        void update_from_arrow_array(std::shared_ptr<arrow::Array> src) {
            std::shared_ptr<arrow::PrimitiveArray> src_arr =
                std::dynamic_pointer_cast<arrow::PrimitiveArray, arrow::Array>(src);
            assert(src_arr);

            T* dst = raw_slice();
            T* typed_src = (T*)src_arr->values()->data();

            for (int64_t i = 0; i < src->length(); i++) {
                dst[i] = typed_src[i];
            }
        }

        void send_rand_1d_cmd() {
            shmem_ml_rand_1d<T> cmd(id);
            send_cmd(&cmd);
        }

        void send_apply_1d_cmd(char* s, int length) {
            shmem_ml_apply_1d<T> *cmd = (shmem_ml_apply_1d<T>*)malloc(
                    sizeof(*cmd) + length);
            assert(cmd);
            new (cmd) shmem_ml_apply_1d<T>(id);
            memcpy(cmd + 1, s, length);

            send_cmd(cmd, sizeof(*cmd) + length);
            free(cmd);
        }

        unsigned get_id() { return id; }

        virtual inline T* raw_slice() = 0;
        virtual std::shared_ptr<arrow::Array> get_arrow_array() = 0;

    protected:
        void setup_psync() {
            for (int i = 0; i < SHMEMML_MAX(SHMEM_REDUCE_SYNC_SIZE, SHMEM_BCAST_SYNC_SIZE); i++) {
                psync[i] = SHMEM_SYNC_VALUE;
            }
            shmem_barrier_all();
        }

        T* pwork;
        long *psync;

#ifdef ATOMICS_AS_MSGS
        void process_atomic_msgs() {
            int success = 1;
            do {
                success = 0;
                T old_val, new_val;
                int msg_len = mailbox_recv(buffered_atomics,
                        MAX_BUFFERED_ATOMICS * sizeof(*buffered_atomics),
                        &atomics_mailbox);
                if (msg_len) {
                    assert(msg_len % sizeof(*buffered_atomics) == 0);
                    assert(msg_len <= MAX_BUFFERED_ATOMICS * sizeof(*buffered_atomics));
                    success = 1;

                    size_t nmsgs = msg_len / sizeof(*buffered_atomics);
                    for (unsigned m = 0; m < nmsgs; m++) {
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
                            atomics_cb((ShmemML1D<T>*)this,
                                    _local_slice_start + msg->local_index,
                                    msg->op, old_val, new_val);
                        }
                    }
                }
            } while (success);
        }
#endif

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

        unsigned id;
        int64_t _N;
        int64_t _chunk_size;
        int64_t _local_slice_start;
        int64_t _local_slice_end;
        ShmemMemoryPool* pool;
        T* symm_reduce_dest;
        T* symm_reduce_src;
        shmem_ml_cmd<T> destroy_cmd;

#ifdef ATOMICS_AS_MSGS
        mailbox_t atomics_mailbox;
        mailbox_buffer_t atomics_mailbox_buffer;
        atomics_msg_t<T> *buffered_atomics;
        int n_done_pes;
        atomics_msg_1d_result_handler<T> atomics_cb;
#endif
};

template<typename T>
class ShmemML1D : public ShmemML1D_Base<T> {
    public:
        ShmemML1D(int64_t N, T init_val,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_1d_result_handler<T> _atomics_cb = NULL
#endif
                ) : ShmemML1D_Base<T>(N, max_shmem_reduction_n
#ifdef ATOMICS_AS_MSGS
                    , _atomics_cb
#endif
                    ) {

            std::shared_ptr<arrow::DataType> type = arrow::fixed_size_binary(sizeof(T));

            arrow::FixedSizeBinaryBuilder builder(type, this->pool);
            // CHECK_ARROW(builder.AppendNulls(_chunk_size));
            CHECK_ARROW(builder.Reserve(this->_chunk_size));
            for (int64_t i = 0; i < this->_chunk_size; i++) {
                CHECK_ARROW(builder.Append((const uint8_t*)&init_val));
            }
            builder.Finish(&_arr);
            _arr->ValidateFull();
        }

        std::shared_ptr<arrow::Array> get_arrow_array() override {
            return std::dynamic_pointer_cast<arrow::Array,
                   arrow::FixedSizeBinaryArray>(_arr);
        }

        inline T* raw_slice() override {
            return (T *)_arr->values()->data();
        }

        void atomic_add(int64_t global_index, T val) override {
            throw std::runtime_error("atomic_add unsupported");
        }

        T atomic_fetch_add(int64_t global_index, T val) override {
            throw std::runtime_error("atomic_fetch_add unsupported");
        }

        T atomic_cas(int64_t global_index, T expected, T update_to) override {
            throw std::runtime_error("atomic_cas unsupported");
        }

        void atomic_or(int64_t global_index, T mask) override {
            throw std::runtime_error("atomic_or unsupported");
        }

        T max(T min_val) override {
            throw std::runtime_error("max unsupported");
        }

        T sum(T zero_val) override {
            throw std::runtime_error("sum unsupported");
        }

    private:
        std::shared_ptr<arrow::FixedSizeBinaryArray> _arr;
};

template<>
class ShmemML1D<long long> : public ShmemML1D_Base<long long> {
    public:
        ShmemML1D(int64_t N,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_1d_result_handler<long long> _atomics_cb = NULL
#endif
                ) : ShmemML1D_Base(N,
                    shmem_ml_create_1d_longlong(N),
                    shmem_ml_destroy_1d_longlong(id_counter),
                    max_shmem_reduction_n
#ifdef ATOMICS_AS_MSGS
                    , _atomics_cb
#endif
                    ) {
            assert(sizeof(long long) == sizeof(int64_t));
            arrow::NumericBuilder<arrow::Int64Type> builder(arrow::int64(), pool);
            // CHECK_ARROW(builder.AppendNulls(_chunk_size));
            CHECK_ARROW(builder.Reserve(_chunk_size));
            for (int64_t i = 0; i < _chunk_size; i++) {
                CHECK_ARROW(builder.Append(0));
            }
            builder.Finish(&_arr);
            _arr->ValidateFull();
        }

        std::shared_ptr<arrow::Array> get_arrow_array() override {
            return std::dynamic_pointer_cast<arrow::Array,
                   arrow::NumericArray<arrow::Int64Type>>(_arr);
        }

        inline long long* raw_slice() override {
            return (long long *)_arr->values()->data();
        }

        void atomic_add(int64_t global_index, long long val) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_longlong_atomic_add(raw_slice() + offset, val, pe);
        }

        long long atomic_fetch_add(int64_t global_index, long long val) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            return shmem_longlong_atomic_fetch_add(raw_slice() + offset, val, pe);
        }

        long long atomic_cas(int64_t global_index, long long expected, long long update_to) override {
            throw std::runtime_error("long long unsupported");
        }

        void atomic_or(int64_t global_index, long long mask) override {
            throw std::runtime_error("long long unsupported");
        }

        long long max(long long min_val) override {
            long long my_max = min_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                if (raw_slice()[i] > my_max) {
                    my_max = raw_slice()[i];
                }
            }

            *symm_reduce_src = my_max;
            setup_psync();
            shmem_longlong_max_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

        long long sum(long long zero_val) override {
            long long my_sum = zero_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                my_sum += raw_slice()[i];
            }

            *symm_reduce_src = my_sum;
            setup_psync();
            shmem_longlong_sum_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

    private:
        std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> _arr;
};

template<>
class ShmemML1D<int64_t> : public ShmemML1D_Base<int64_t> {
    public:
        ShmemML1D(int64_t N,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_1d_result_handler<int64_t> _atomics_cb = NULL
#endif
                ) : ShmemML1D_Base(N,
                    shmem_ml_create_1d_int64(N),
                    shmem_ml_destroy_1d_int64(id_counter),
                    max_shmem_reduction_n
#ifdef ATOMICS_AS_MSGS
                    , _atomics_cb
#endif
                    ) {
            arrow::NumericBuilder<arrow::Int64Type> builder(arrow::int64(), pool);
            // CHECK_ARROW(builder.AppendNulls(_chunk_size));
            CHECK_ARROW(builder.Reserve(_chunk_size));
            for (int64_t i = 0; i < _chunk_size; i++) {
                CHECK_ARROW(builder.Append(0));
            }
            builder.Finish(&_arr);
            _arr->ValidateFull();
        }

        std::shared_ptr<arrow::Array> get_arrow_array() override {
            return std::dynamic_pointer_cast<arrow::Array,
                   arrow::NumericArray<arrow::Int64Type>>(_arr);
        }

        inline int64_t* raw_slice() override {
            return (int64_t *)_arr->values()->data();
        }

        void atomic_add(int64_t global_index, int64_t val) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_int64_atomic_add(raw_slice() + offset, val, pe);
        }

        int64_t atomic_fetch_add(int64_t global_index, int64_t val) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            return shmem_int64_atomic_fetch_add(raw_slice() + offset, val, pe);
        }

        int64_t atomic_cas(int64_t global_index, int64_t expected, int64_t update_to) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            return shmem_int64_atomic_compare_swap(raw_slice() + offset, expected, update_to, pe);
        }

        void atomic_or(int64_t global_index, int64_t mask) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_int64_atomic_or(raw_slice() + offset, mask, pe);
        }

        int64_t max(int64_t min_val) override {
            assert(sizeof(int64_t) == sizeof(long));
            int64_t my_max = min_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                if (raw_slice()[i] > my_max) {
                    my_max = raw_slice()[i];
                }
            }

            *symm_reduce_src = my_max;
            setup_psync();
            shmem_long_max_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

        int64_t sum(int64_t zero_val) override {
            assert(sizeof(int64_t) == sizeof(long));
            int64_t my_sum = zero_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                my_sum += raw_slice()[i];
            }

            *symm_reduce_src = my_sum;
            setup_psync();
            shmem_long_sum_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

    private:
        std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> _arr;
};

template<>
class ShmemML1D<uint32_t> : public ShmemML1D_Base<uint32_t> {
    public:
        ShmemML1D(int64_t N,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_1d_result_handler<uint32_t> _atomics_cb = NULL
#endif
                ) : ShmemML1D_Base(N,
                    shmem_ml_create_1d_uint32(N),
                    shmem_ml_destroy_1d_uint32(id_counter),
                    max_shmem_reduction_n
#ifdef ATOMICS_AS_MSGS
                    , _atomics_cb
#endif
                    ) {
            arrow::NumericBuilder<arrow::UInt32Type> builder(arrow::uint32(), pool);
            // CHECK_ARROW(builder.AppendNulls(_chunk_size));
            CHECK_ARROW(builder.Reserve(_chunk_size));
            for (int64_t i = 0; i < _chunk_size; i++) {
                CHECK_ARROW(builder.Append(0));
            }
            builder.Finish(&_arr);
            _arr->ValidateFull();
        }

        std::shared_ptr<arrow::Array> get_arrow_array() override {
            return std::dynamic_pointer_cast<arrow::Array,
                   arrow::NumericArray<arrow::UInt32Type>>(_arr);
        }

        inline uint32_t* raw_slice() override {
            return (uint32_t *)_arr->values()->data();
        }

        void atomic_add(int64_t global_index, uint32_t val) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_uint32_atomic_add(raw_slice() + offset, val, pe);
        }

        uint32_t atomic_fetch_add(int64_t global_index, uint32_t val) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            return shmem_uint32_atomic_fetch_add(raw_slice() + offset, val, pe);
        }

        uint32_t atomic_cas(int64_t global_index, uint32_t expected, uint32_t update_to) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            return shmem_uint32_atomic_compare_swap(raw_slice() + offset, expected, update_to, pe);
        }

        void atomic_or(int64_t global_index, uint32_t mask) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            shmem_uint32_atomic_or(raw_slice() + offset, mask, pe);
        }

        uint32_t max(uint32_t min_val) override {
            uint32_t my_max = min_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                if (raw_slice()[i] > my_max) {
                    my_max = raw_slice()[i];
                }
            }

            *symm_reduce_src = my_max;
            setup_psync();
            shmem_int_max_to_all((int*)symm_reduce_dest, (int*)symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), (int*)pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

        uint32_t sum(uint32_t zero_val) override {
            uint32_t my_sum = zero_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                my_sum += raw_slice()[i];
            }

            *symm_reduce_src = my_sum;
            setup_psync();
            shmem_int_sum_to_all((int*)symm_reduce_dest, (int*)symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), (int*)pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

    private:
        std::shared_ptr<arrow::NumericArray<arrow::UInt32Type>> _arr;
};

template<>
class ShmemML1D<double> : public ShmemML1D_Base<double> {
    public:
        ShmemML1D(int64_t N,
                unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_1d_result_handler<double> _atomics_cb = NULL
#endif
                ) : ShmemML1D_Base(N,
                    shmem_ml_create_1d_double(N),
                    shmem_ml_destroy_1d_double(id_counter),
                    max_shmem_reduction_n
#ifdef ATOMICS_AS_MSGS
                    , _atomics_cb
#endif
                    ) {
            arrow::NumericBuilder<arrow::DoubleType> builder(arrow::float64(), pool);
            // CHECK_ARROW(builder.AppendNulls(_chunk_size));
            CHECK_ARROW(builder.Reserve(_chunk_size));
            for (int64_t i = 0; i < _chunk_size; i++) {
                CHECK_ARROW(builder.Append(0));
            }
            builder.Finish(&_arr);
            _arr->ValidateFull();
        }

        std::shared_ptr<arrow::Array> get_arrow_array() override {
            return std::dynamic_pointer_cast<arrow::Array,
                   arrow::NumericArray<arrow::DoubleType>>(_arr);
        }

        inline double* raw_slice() override {
            return (double *)_arr->values()->data();
        }

        void atomic_add(int64_t global_index, double val) override {
            atomic_fetch_add(global_index, val);
        }

        double atomic_fetch_add(int64_t global_index, double val) override {
            double old_val = 0.0;
            double expected, new_val;

            do {
                expected = old_val;
                new_val = expected + val;
                old_val = atomic_cas(global_index, expected, new_val);
            } while (old_val != expected);
            return old_val;
        }

        double atomic_cas(int64_t global_index, double expected, double update_to) override {
            int pe = global_index / _chunk_size;
            int64_t offset = global_index % _chunk_size;
            double *ptr = raw_slice() + offset;

            union {
                int64_t i;
                double d;
            } _expected, _new_val, _old_val;

            _expected.d = expected;
            _new_val.d = update_to;
            _old_val.i = shmem_int64_atomic_compare_swap((int64_t*)ptr,
                    _expected.i, _new_val.i, pe);

            return _old_val.d;
        }

        void atomic_or(int64_t global_index, double mask) override {
            throw std::runtime_error("double unsupported");
        }

        double max(double min_val) override {
            double my_max = min_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                if (raw_slice()[i] > my_max) {
                    my_max = raw_slice()[i];
                }
            }

            *symm_reduce_src = my_max;
            setup_psync();
            shmem_double_max_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

        double sum(double zero_val) override {
            double my_sum = zero_val;
            for (int64_t i = 0; i < local_slice_end() - local_slice_start(); i++) {
                my_sum += raw_slice()[i];
            }

            *symm_reduce_src = my_sum;
            setup_psync();
            shmem_double_sum_to_all(symm_reduce_dest, symm_reduce_src, 1, 0, 0,
                    shmem_n_pes(), pwork, psync);
            shmem_barrier_all();
            return *symm_reduce_dest;
        }

    private:
        std::shared_ptr<arrow::NumericArray<arrow::DoubleType>> _arr;
};

class ShmemML2D {
    public:
        ShmemML2D(int64_t M, int64_t N, unsigned max_shmem_reduction_n = 1
#ifdef ATOMICS_AS_MSGS
                , atomics_msg_2d_result_handler<double> _atomics_cb = NULL
#endif
                ) {
            _M = M;
            _N = N;
#ifdef ATOMICS_AS_MSGS
            atomics_cb = _atomics_cb;
#endif
            int npes = shmem_n_pes();
            _rows_per_pe = (_M + npes - 1) / npes;
            id = id_counter++;

            shmem_ml_create_2d cmd(M, N);
            send_cmd(&cmd);
            shmem_barrier_all();

            pool = ShmemMemoryPool::get();

            // Construct a simple schema for this table
            std::vector<std::shared_ptr<arrow::Field>> fields;
            for (int i = 0; i < N; i++) {
                std::stringstream ss;
                ss << i;
                fields.push_back(arrow::field(ss.str(), arrow::float64()));
            }
            std::shared_ptr<arrow::Schema> schema = arrow::schema(fields);

            std::vector<std::shared_ptr<arrow::Array>> arrs;
            std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
            for (int i = 0; i < N; i++) {
                std::shared_ptr<arrow::NumericArray<arrow::DoubleType>> arr;

                arrow::NumericBuilder<arrow::DoubleType> builder(
                        arrow::float64(), pool);
                CHECK_ARROW(builder.Reserve(_rows_per_pe));
                for (int64_t i = 0; i < _rows_per_pe; i++) {
                    CHECK_ARROW(builder.Append(0));
                }
                builder.Finish(&arr);
                arr->ValidateFull();
                arrs.push_back(arr);

                columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
            }

            _arrs = arrow::Table::Make(schema, columns, _rows_per_pe);
            _arrs->Validate();
            shmem_barrier_all();
            add_array_to_namespace(id, this);
            end_cmd();
        }

        ~ShmemML2D() {
            shmem_ml_destroy_2d cmd(id);
            send_cmd(&cmd);

            // Collective destruction
            shmem_barrier_all();
            end_cmd();
        }

        int64_t N() { return _N; }
        int64_t M() { return _M; }
        int64_t rows_per_pe() { return _rows_per_pe; }
        void sync() {
            shmem_ml_sync_2d<double> cmd(id);
            send_cmd(&cmd);

            shmem_barrier_all();
            end_cmd();
        }

        void clear(double v) {
            for (int c = 0; c < _arrs->num_columns(); c++) {
                double *symm = get_column_array(c);

                if (v == 0.0) {
                    memset(symm, 0x00, _rows_per_pe * sizeof(*symm));
                } else {
                    for (int i = 0; i < _rows_per_pe; i++) {
                        symm[i] = v;
                    }
                }
            }
        }

        void set(int64_t row, int64_t col, double v) {
            double* symm = get_column_array(col);
            int pe = row / _rows_per_pe;
            int64_t offset = row % _rows_per_pe;
            shmem_putmem(symm + offset, &v, sizeof(v), pe);
        }

        double get(int64_t row, int64_t col) {
            double* symm = get_column_array(col);
            int pe = row / _rows_per_pe;
            int64_t offset = row % _rows_per_pe;

            double val;
            shmem_getmem(&val, symm + offset, sizeof(val), pe);
            return val;
        }

        std::shared_ptr<arrow::Table> get_arrow_table() {
            return _arrs;
        }

        void update_from_arrow_table(std::shared_ptr<arrow::Table> src) {
            assert(src->num_columns() == _arrs->num_columns());
            for (int c = 0; c < src->num_columns(); c++) {
                assert(_arrs->column(c)->length() == _arrs->column(c)->chunk(0)->length());
                assert(src->column(c)->length() == src->column(c)->chunk(0)->length());

                std::shared_ptr<arrow::Array> dst_arr = _arrs->column(c)->chunk(0);
                std::shared_ptr<arrow::Array> src_arr = src->column(c)->chunk(0);
                assert(src_arr->length() == dst_arr->length());

                std::shared_ptr<arrow::NumericArray<arrow::DoubleType>> dst_prim_arr =
                    std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>, arrow::Array>(dst_arr);
                assert(dst_prim_arr);
                double* dst_symm = (double*)dst_prim_arr->values()->data();

                std::shared_ptr<arrow::NumericArray<arrow::DoubleType>> dbl_src_prim_arr =
                    std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>, arrow::Array>(src_arr);
                std::shared_ptr<arrow::NumericArray<arrow::FloatType>> flt_src_prim_arr =
                    std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>, arrow::Array>(src_arr);

                if (dbl_src_prim_arr) {
                    double* src_symm = (double*)dbl_src_prim_arr->values()->data();
                    for (int i = 0; i < src_arr->length(); i++) {
                        dst_symm[i] = src_symm[i];
                    }
                } else if (flt_src_prim_arr) {
                    float* src_symm = (float*)flt_src_prim_arr->values()->data();
                    for (int i = 0; i < src_arr->length(); i++) {
                        dst_symm[i] = src_symm[i];
                    }
                } else {
                    abort();
                }
            }
        }

        unsigned get_id() { return id; }

        void send_rand_2d_cmd() {
            shmem_ml_rand_2d<double> cmd(id);
            send_cmd(&cmd);
        }

        void send_apply_2d_cmd(char* s, int length) {
            shmem_ml_apply_2d<double> *cmd = (shmem_ml_apply_2d<double>*)malloc(
                    sizeof(*cmd) + length);
            assert(cmd);
            new (cmd) shmem_ml_apply_2d<double>(id);
            memcpy(cmd + 1, s, length);

            send_cmd(cmd, sizeof(*cmd) + length);
            free(cmd);
        }

    private:

        inline double* get_column_array(int64_t col) {
            assert(col < _arrs->num_columns());
            std::shared_ptr<arrow::ChunkedArray> col_chunked = _arrs->column(
                    col);
            std::shared_ptr<arrow::Array> col_arr = col_chunked->chunk(0);
            assert(col_chunked->length() == col_arr->length());

            std::shared_ptr<arrow::PrimitiveArray> src_arr =
                std::dynamic_pointer_cast<arrow::PrimitiveArray, arrow::Array>(
                        col_arr);
            assert(src_arr);

            double* symm = (double*)src_arr->values()->data();
            return symm;
        }

        std::shared_ptr<arrow::Table> _arrs;
        int64_t _M, _N;
        int64_t _rows_per_pe;
        unsigned id;
        ShmemMemoryPool* pool;
#ifdef ATOMICS_AS_MSGS
        atomics_msg_2d_result_handler<double> atomics_cb;
#endif
};

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
        void reduce_all_sum();
        void bcast(int src_rank);

        void zero() {
            memset(this->raw_slice(), 0x00, _replicated_N * sizeof(T));
            shmem_barrier_all();
        }
};

template<>
void ReplicatedShmemML1D<int>::reduce_all_or();

template<>
void ReplicatedShmemML1D<unsigned>::reduce_all_or();

template<>
void ReplicatedShmemML1D<unsigned>::reduce_all_sum();

template<>
void ReplicatedShmemML1D<unsigned>::bcast(int src_rank);

template<>
void ReplicatedShmemML1D<double>::reduce_all_sum();

template<>
void ReplicatedShmemML1D<double>::bcast(int src_rank);

void shmem_ml_init();
void shmem_ml_finalize();
bool is_client_server_mode();

unsigned long long shmem_ml_current_time_us();

template <typename T>
shmem_ml_py_cmd cmd_handler(shmem_ml_command cmd, void* _payload,
        size_t payload_size) {
    shmem_ml_py_cmd optional_cmd;

    shmem_ml_cmd_payload<T> *payload = (shmem_ml_cmd_payload<T>*)_payload;
    assert(payload_size >= sizeof(*payload));

    switch (cmd) {
        case (CREATE_1D): {
            ShmemML1D<T> *arr = new ShmemML1D<T>(payload->create_1d.N);
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d CREATE_1D id=%u\n", shmem_my_pe(), arr->get_id());
#endif
            break;
        }
        case (DESTROY_1D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d DESTROY_1D id=%u\n", shmem_my_pe(), payload->destroy_1d.id);
#endif
            ShmemML1D<T>* arr = (ShmemML1D<T>*)delete_array_in_namespace(payload->destroy_1d.id);
            delete arr;
            break;
        }
        case (CREATE_2D): {
            ShmemML2D *arr = new ShmemML2D(payload->create_2d.M,
                    payload->create_2d.N);
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d CREATE_2D id=%u\n", shmem_my_pe(), arr->get_id());
#endif
            break;
        }
        case (DESTROY_2D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d DESTROY_2D id=%u\n", shmem_my_pe(), payload->destroy_2d.id);
#endif
            ShmemML2D* arr = (ShmemML2D*)delete_array_in_namespace(payload->destroy_2d.id);
            delete arr;
            break;
        }
        case (CLEAR_1D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d CLEAR_1D\n", shmem_my_pe());
#endif
            ShmemML1D<T>* arr = (ShmemML1D<T>*)lookup_array_in_namespace(payload->clear_1d.id);
            arr->clear(payload->clear_1d.val);
            break;
        }
        case (SYNC_1D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d SYNC_1D\n", shmem_my_pe());
#endif
            ShmemML1D<T>* arr = (ShmemML1D<T>*)lookup_array_in_namespace(payload->sync_1d.id);
            arr->sync();
            break;
        }
        case (SYNC_2D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d SYNC_2D\n", shmem_my_pe());
#endif
            ShmemML2D* arr = (ShmemML2D*)lookup_array_in_namespace(payload->sync_2d.id);
            arr->sync();
            break;
        }
        case (RAND_1D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d RAND_1D id=%u\n", shmem_my_pe(), payload->rand_1d.id);
#endif
            ShmemML1D<T>* arr = (ShmemML1D<T>*)lookup_array_in_namespace(payload->rand_1d.id);
            optional_cmd = shmem_ml_py_cmd(RAND_1D, (void*)arr, NULL, 0, NULL);
            break;
        }
        case (RAND_2D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d RAND_2D id=%u\n", shmem_my_pe(),
                    payload->rand_2d.id);
#endif
            ShmemML2D* arr = (ShmemML2D*)lookup_array_in_namespace(payload->rand_2d.id);
            optional_cmd = shmem_ml_py_cmd(RAND_2D, (void*)arr, NULL, 0, NULL);
            break;
        }
        case (APPLY_1D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d APPLY_1D id=%u\n", shmem_my_pe(),
                    payload->apply_1d.id);
#endif
            ShmemML1D<T>* arr = (ShmemML1D<T>*)lookup_array_in_namespace(payload->apply_1d.id);
            size_t serialized_func_length = payload_size - sizeof(*payload);

            char *serialized_func = (char *)malloc(serialized_func_length + 1);
            assert(serialized_func);
            memcpy(serialized_func, payload + 1, serialized_func_length);
            serialized_func[serialized_func_length] = '\0';

            optional_cmd = shmem_ml_py_cmd(APPLY_1D, (void*)arr,
                    serialized_func, serialized_func_length, NULL);
            break;
        }
        case (APPLY_2D): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d APPLY_2D id=%u\n", shmem_my_pe(),
                    payload->apply_2d.id);
#endif
            ShmemML2D* arr = (ShmemML2D*)lookup_array_in_namespace(payload->apply_2d.id);
            size_t serialized_func_length = payload_size - sizeof(*payload);

            char *serialized_func = (char *)malloc(serialized_func_length + 1);
            assert(serialized_func);
            memcpy(serialized_func, payload + 1, serialized_func_length);
            serialized_func[serialized_func_length] = '\0';

            optional_cmd = shmem_ml_py_cmd(APPLY_2D, (void*)arr,
                    serialized_func, serialized_func_length, NULL);
            break;
        }
        case (SGD_FIT): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d SGD_FIT x_id=%u y_id=%u\n", shmem_my_pe(),
                    payload->sgd_fit.x_id, payload->sgd_fit.y_id);
#endif
            unsigned x_id = payload->sgd_fit.x_id;
            unsigned y_id = payload->sgd_fit.y_id;
            ShmemML2D* x_arr = (ShmemML2D*)lookup_array_in_namespace(x_id);
            ShmemML1D<double>* y_arr = (ShmemML1D<double>*)lookup_array_in_namespace(y_id);

            size_t serialized_model_length = payload_size - sizeof(*payload);

            char *serialized_model = (char *)malloc(serialized_model_length + 1);
            assert(serialized_model);
            memcpy(serialized_model, payload + 1, serialized_model_length);
            serialized_model[serialized_model_length] = '\0';

            optional_cmd = shmem_ml_py_cmd(SGD_FIT, (void*)x_arr,
                    serialized_model, serialized_model_length, (void*)y_arr);
            break;
        }
        case (SGD_PREDICT): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d SGD_PREDICT x_id=%u\n", shmem_my_pe(),
                    payload->sgd_predict.x_id);
#endif
            unsigned x_id = payload->sgd_predict.x_id;
            ShmemML2D* x_arr = (ShmemML2D*)lookup_array_in_namespace(x_id);

            size_t serialized_model_length = payload_size - sizeof(*payload);

            char *serialized_model = (char *)malloc(serialized_model_length + 1);
            assert(serialized_model);
            memcpy(serialized_model, payload + 1, serialized_model_length);
            serialized_model[serialized_model_length] = '\0';

            optional_cmd = shmem_ml_py_cmd(SGD_PREDICT, (void*)x_arr,
                    serialized_model, serialized_model_length, NULL);
            break;
        }
        case (SEQUENTIAL_FIT): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d SEQUENTIAL_FIT x_id=%u y_id=%u\n", shmem_my_pe(),
                    payload->sequential_fit.x_id, payload->sequential_fit.y_id);
#endif
            unsigned x_id = payload->sequential_fit.x_id;
            unsigned y_id = payload->sequential_fit.y_id;
            ShmemML2D* x_arr = (ShmemML2D*)lookup_array_in_namespace(x_id);
            ShmemML1D<double>* y_arr = (ShmemML1D<double>*)lookup_array_in_namespace(y_id);

            size_t serialized_model_length = payload_size - sizeof(*payload);

            char *serialized_model = (char *)malloc(serialized_model_length + 1);
            assert(serialized_model);
            memcpy(serialized_model, payload + 1, serialized_model_length);
            serialized_model[serialized_model_length] = '\0';

            optional_cmd = shmem_ml_py_cmd(SEQUENTIAL_FIT, (void*)x_arr,
                    serialized_model, serialized_model_length, (void*)y_arr);
            break;
        }
        case (SEQUENTIAL_PREDICT): {
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d SEQUENTIAL_PREDICT x_id=%u\n", shmem_my_pe(),
                    payload->sequential_predict.x_id);
#endif
            unsigned x_id = payload->sequential_predict.x_id;
            ShmemML2D* x_arr = (ShmemML2D*)lookup_array_in_namespace(x_id);

            size_t serialized_model_length = payload_size - sizeof(*payload);

            char *serialized_model = (char *)malloc(serialized_model_length + 1);
            assert(serialized_model);
            memcpy(serialized_model, payload + 1, serialized_model_length);
            serialized_model[serialized_model_length] = '\0';

            optional_cmd = shmem_ml_py_cmd(SEQUENTIAL_PREDICT, (void*)x_arr,
                    serialized_model, serialized_model_length, NULL);
            break;
        }
        case (CMD_DONE):
#ifdef VERBOSE_CMD
            fprintf(stderr, "PE %d CMD_DONE\n", shmem_my_pe());
#endif
            optional_cmd = shmem_ml_py_cmd(CMD_DONE, NULL, NULL, 0, NULL);
            break;
        default:
#ifdef VERBOSE_CMD
            fprintf(stderr, "ERROR: Unexpected command %d\n", cmd);
#endif
            abort();
    }
    return optional_cmd;
}

/*
 * For native C/C++ applications.
 */
template <typename lambda>
static inline void shmem_ml_client_server_launch(lambda l) {
    if (getenv("SHMEM_ML_HANG_ABORT")) {
        pthread_t aborting_pthread;
        const int pthread_err = pthread_create(&aborting_pthread, NULL,
                aborting_thread, NULL);
        assert(pthread_err == 0);
    }

    bool is_server = setup_client_server();

    mailbox_init(&cmd_mailbox, 32 * 1024 * 1024);
    cmd_msg = mailbox_allocate_msg(MAX_CMD_LEN, &cmd_ctx);

    if (is_server) {
        command_loop();
    } else {
        l();

        shmem_ml_cmd_done done_cmd;
        send_cmd(&done_cmd);
        end_cmd();
    }

    mailbox_destroy(&cmd_mailbox);
    this_pe_has_exited = 1;
}

/*
 * For Python applications.
 */
static inline void py_shmem_ml_client_server_launch(const char *python_file) {
    if (getenv("SHMEM_ML_HANG_ABORT")) {
        pthread_t aborting_pthread;
        const int pthread_err = pthread_create(&aborting_pthread, NULL,
                aborting_thread, NULL);
        assert(pthread_err == 0);
    }

    bool is_server = setup_client_server();

    mailbox_init(&cmd_mailbox, 32 * 1024 * 1024);
    cmd_msg = mailbox_allocate_msg(MAX_CMD_LEN, &cmd_ctx);

    Py_Initialize();
    if (is_server) {
        PyRun_SimpleString("from PyShmemML import shmem_ml_command_loop; shmem_ml_command_loop();");
        Py_Finalize();
    } else {
        FILE *fp = fopen(python_file, "r");
        PyRun_SimpleFileExFlags(fp, python_file, 1, NULL);
        Py_Finalize();

        shmem_ml_cmd_done done_cmd;
        send_cmd(&done_cmd);
        end_cmd();
    }

    mailbox_destroy(&cmd_mailbox);
    this_pe_has_exited = 1;
}


#endif
