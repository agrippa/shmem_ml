#ifndef SHMEM_ML_CMD_HPP
#define SHMEM_ML_CMD_HPP

#include "mailbox.hpp"

#include <map>

#define MAX_CMD_LEN 2048

class shmem_ml_py_cmd;

extern int inside_cmd;
extern mailbox_msg_header_t *cmd_msg;
extern bool client_server_mode;
extern bool is_server;
extern shmem_ctx_t cmd_ctx;
extern mailbox_t cmd_mailbox;

void end_cmd();

typedef enum {
    CREATE_1D,
    DESTROY_1D,
    CREATE_2D,
    DESTROY_2D,
    CLEAR_1D,
    SYNC_1D,
    GET_1D,
    RAND_1D,
    RAND_2D,
    APPLY_1D,
    CMD_DONE,
    CMD_INVALID
} shmem_ml_command;

template <typename T>
union shmem_ml_cmd_payload {
    struct {
        int64_t N;
    } create_1d;
    struct {
        unsigned id;
    } destroy_1d;
    struct {
        int64_t M, N;
    } create_2d;
    struct {
        unsigned id;
    } destroy_2d;
    struct {
        unsigned id;
        T val;
    } clear_1d;
    struct {
        unsigned id;
    } sync_1d;
    struct {
        unsigned id;
    } rand_1d;
    struct {
        unsigned id;
    } rand_2d;
    struct {
        unsigned id;
    } apply_1d;
};

template <typename T>
shmem_ml_py_cmd cmd_handler(shmem_ml_command cmd, void* _payload,
        size_t payload_size);

typedef shmem_ml_py_cmd (*cmd_handler_ptr)(shmem_ml_command, void*, size_t);

template <typename T>
cmd_handler_ptr get_typed_cmd_handler() {
    return NULL;
}

/*
 * Needs to be declared here and implemented in shmem_ml.cpp for each type that
 * might be used from a Python program. Because the Python module consists of
 * its own .so (in addition to libshmem_ml.so), there are duplicate definitions
 * in the two .so files of everything in the header (including things like
 * cmd_handler<double>). This causes problems when PE 0 (which loads the Python
 * module's .so) sends the address of some cmd_handler<T> to the other PEs,
 * which haven't loaded the Python .so and its symbols.
 */
template <> cmd_handler_ptr get_typed_cmd_handler<double>();

template <typename T>
struct shmem_ml_cmd {
    shmem_ml_command cmd;
    cmd_handler_ptr handler;
    shmem_ml_cmd_payload<T> payload;

    shmem_ml_cmd(shmem_ml_command _cmd) {
        cmd = _cmd;

        handler = get_typed_cmd_handler<T>();
        if (handler == NULL) {
            handler = cmd_handler<T>;
        }
    }
};

struct shmem_ml_create_1d_double : public shmem_ml_cmd<double> {
    shmem_ml_create_1d_double(int64_t N) : shmem_ml_cmd<double>(CREATE_1D) {
        payload.create_1d.N = N;
    }
};

struct shmem_ml_destroy_1d_double : public shmem_ml_cmd<double> {
    shmem_ml_destroy_1d_double(unsigned id) : shmem_ml_cmd<double>(DESTROY_1D) {
        payload.destroy_1d.id = id;
    }
};

struct shmem_ml_create_1d_longlong : public shmem_ml_cmd<long long> {
    shmem_ml_create_1d_longlong(int64_t N) : shmem_ml_cmd<long long>(CREATE_1D) {
        payload.create_1d.N = N;
    }
};

struct shmem_ml_destroy_1d_longlong : public shmem_ml_cmd<long long> {
    shmem_ml_destroy_1d_longlong(unsigned id) : shmem_ml_cmd<long long>(DESTROY_1D) {
        payload.destroy_1d.id = id;
    }
};

struct shmem_ml_create_1d_int64 : public shmem_ml_cmd<int64_t> {
    shmem_ml_create_1d_int64(int64_t N) : shmem_ml_cmd<int64_t>(CREATE_1D) {
        payload.create_1d.N = N;
    }
};

struct shmem_ml_destroy_1d_int64 : public shmem_ml_cmd<int64_t> {
    shmem_ml_destroy_1d_int64(unsigned id) : shmem_ml_cmd<int64_t>(DESTROY_1D) {
        payload.destroy_1d.id = id;
    }
};

struct shmem_ml_create_1d_uint32 : public shmem_ml_cmd<uint32_t> {
    shmem_ml_create_1d_uint32(int64_t N) : shmem_ml_cmd<uint32_t>(CREATE_1D) {
        payload.create_1d.N = N;
    }
};

struct shmem_ml_destroy_1d_uint32 : public shmem_ml_cmd<uint32_t> {
    shmem_ml_destroy_1d_uint32(unsigned id) : shmem_ml_cmd<uint32_t>(DESTROY_1D) {
        payload.destroy_1d.id = id;
    }
};

struct shmem_ml_create_2d : public shmem_ml_cmd<double> {
    shmem_ml_create_2d(int64_t M, int64_t N) : shmem_ml_cmd<double>(CREATE_2D) {
        payload.create_2d.M = M;
        payload.create_2d.N = N;
    }
};

struct shmem_ml_destroy_2d : public shmem_ml_cmd<double> {
    shmem_ml_destroy_2d(unsigned id) : shmem_ml_cmd<double>(DESTROY_2D) {
        payload.destroy_2d.id = id;
    }
};

template <typename T>
struct shmem_ml_clear_1d : public shmem_ml_cmd<T> {
    shmem_ml_clear_1d(unsigned id, T val) : shmem_ml_cmd<T>(CLEAR_1D) {
        this->payload.clear_1d.id = id;
        this->payload.clear_1d.val = val;
    }
};

template <typename T>
struct shmem_ml_sync_1d : public shmem_ml_cmd<T> {
    shmem_ml_sync_1d(unsigned id) : shmem_ml_cmd<T>(SYNC_1D) {
        this->payload.sync_1d.id = id;
    }
};

template <typename T>
struct shmem_ml_rand_1d : public shmem_ml_cmd<T> {
    shmem_ml_rand_1d(unsigned id) : shmem_ml_cmd<T>(RAND_1D) {
        this->payload.rand_1d.id = id;
    }
};

template <typename T>
struct shmem_ml_apply_1d : public shmem_ml_cmd<T> {
    shmem_ml_apply_1d(unsigned id) : shmem_ml_cmd<T>(APPLY_1D) {
        this->payload.apply_1d.id = id;
    }
};

template <typename T>
struct shmem_ml_rand_2d : public shmem_ml_cmd<T> {
    shmem_ml_rand_2d(unsigned id) : shmem_ml_cmd<T>(RAND_2D) {
        this->payload.rand_2d.id = id;
    }
};

struct shmem_ml_cmd_done : public shmem_ml_cmd<uint32_t> {
    shmem_ml_cmd_done() : shmem_ml_cmd<uint32_t>(CMD_DONE) {
    }
};

template<typename T>
static inline void send_cmd(shmem_ml_cmd<T>* cmd, int _cmd_len = -1) {
    inside_cmd++;
    if (client_server_mode && !is_server && inside_cmd == 1) {
        size_t cmd_len = sizeof(*cmd);
        if (_cmd_len > 0) {
            cmd_len = _cmd_len;
        }
        assert(cmd_len <= MAX_CMD_LEN);

        memcpy(cmd_msg + 1, cmd, cmd_len);

        for (int p = 1; p < shmem_n_pes(); p++) {
            int success = 0;
            while (!success) {
                success = mailbox_send(cmd_msg, cmd_ctx, cmd_len, p, -1,
                        &cmd_mailbox);
            }
        }
    }
}

#endif
