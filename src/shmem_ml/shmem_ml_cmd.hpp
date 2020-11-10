#ifndef SHMEM_ML_CMD_HPP
#define SHMEM_ML_CMD_HPP

#include "mailbox.hpp"

#include <map>

#define MAX_CMD_LEN 2048

extern mailbox_msg_header_t *cmd_msg;
extern bool client_server_mode;
extern bool is_server;
extern shmem_ctx_t cmd_ctx;
extern mailbox_t cmd_mailbox;

typedef enum {
    CREATE_1D,
    DESTROY_1D,
    CREATE_2D,
    DESTROY_2D,
    CLEAR_1D,
    CMD_DONE,
    CMD_INVALID
} shmem_ml_command;

typedef bool (*cmd_handler_ptr)(shmem_ml_command, void*, size_t, std::map<unsigned, void*>&);

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
};

template <typename T>
bool cmd_handler(shmem_ml_command cmd, void* _payload, size_t payload_size,
        std::map<unsigned, void*>& arrs);

template <typename T>
struct shmem_ml_cmd {
    shmem_ml_command cmd;
    bool (*handler)(shmem_ml_command, void*, size_t, std::map<unsigned, void*>&);
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

struct shmem_ml_cmd_done : public shmem_ml_cmd<uint32_t> {
    shmem_ml_cmd_done() : shmem_ml_cmd<uint32_t>(CMD_DONE) {
    }
};

template<typename T>
static inline void send_cmd(shmem_ml_cmd<T>* cmd) {
    if (client_server_mode && !is_server) {
        assert(sizeof(*cmd) <= MAX_CMD_LEN);

        memcpy(cmd_msg + 1, cmd, sizeof(*cmd));

        for (int p = 1; p < shmem_n_pes(); p++) {
            int success = 0;
            while (!success) {
                success = mailbox_send(cmd_msg, cmd_ctx, sizeof(*cmd),
                        p, -1, &cmd_mailbox);
            }
        }
    }
}


#endif
