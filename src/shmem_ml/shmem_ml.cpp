#include <time.h>
#include <sys/time.h>
#include <map>

#include "ShmemMemoryPool.hpp"
#include "shmem_ml.hpp"

volatile int this_pe_has_exited = 0;

int inside_cmd = 0;
int id_counter = 0;
bool client_server_mode = false;
bool is_server = false;
mailbox_t cmd_mailbox;
mailbox_msg_header_t *cmd_msg;
shmem_ctx_t cmd_ctx;

static std::map<unsigned, void*> arrs;

void add_array_to_namespace(unsigned id, void *arr) {
#ifdef VERBOSE_CMD
    fprintf(stderr, "PE %d add_array_to_namespace id=%u\n", shmem_my_pe(), id);
#endif
    arrs.insert(std::pair<unsigned, void*>(id, arr));
}

void* lookup_array_in_namespace(unsigned id) {
    return arrs.at(id);
}

void* delete_array_in_namespace(unsigned id) {
#ifdef VERBOSE_CMD
    fprintf(stderr, "PE %d delete_array_in_namespace id=%u\n", shmem_my_pe(), id);
#endif
    void* arr = arrs.at(id);
    arrs.erase(id);
    return arr;
}

void send_sgd_fit_cmd(unsigned x_id, unsigned y_id, char *serialized_model,
        int serialized_model_length) {
    shmem_ml_sgd_fit *cmd = (shmem_ml_sgd_fit*)malloc(sizeof(*cmd) + serialized_model_length);
    assert(cmd);
    new (cmd) shmem_ml_sgd_fit(x_id, y_id);
    memcpy(cmd + 1, serialized_model, serialized_model_length);

    send_cmd(cmd, sizeof(*cmd) + serialized_model_length);
    free(cmd);
}

void send_sgd_predict_cmd(unsigned x_id, char *serialized_model,
        int serialized_model_length) {
    shmem_ml_sgd_predict *cmd = (shmem_ml_sgd_predict*)malloc(sizeof(*cmd) + serialized_model_length);
    assert(cmd);
    new (cmd) shmem_ml_sgd_predict(x_id);
    memcpy(cmd + 1, serialized_model, serialized_model_length);

    send_cmd(cmd, sizeof(*cmd) + serialized_model_length);
    free(cmd);
}

template <>
cmd_handler_ptr get_typed_cmd_handler<double>() {
    return cmd_handler<double>;
}

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

void *aborting_thread(void *user_data) {
    UNUSED_VAR(user_data);
    unsigned nseconds = atoi(getenv("SHMEM_ML_HANG_ABORT"));
    assert(nseconds > 0);

    int target_pe = -1;
    if (getenv("SHMEM_ML_HANG_ABORT_PE")) {
        target_pe = atoi(getenv("SHMEM_ML_HANG_ABORT_PE"));
    }

    if (target_pe == -1 || target_pe == shmem_my_pe()) {
        fprintf(stderr, "INFO: SHMEM-ML will forcibly abort PE %d after %d "
                "seconds.\n", shmem_my_pe(), nseconds);

        const unsigned long long start = shmem_ml_current_time_us();
        while (!this_pe_has_exited && shmem_ml_current_time_us() - start < nseconds * 1000000) {
            sleep(10);
        }

        if (!this_pe_has_exited) {
            fprintf(stderr, "INFO: SHMEM-ML forcibly aborting PE %d after %d "
                    "seconds because SHMEM_ML_HANG_ABORT was set.\n", shmem_my_pe(),
                    nseconds);
            abort(); // Get a core dump
        }
    }
    return NULL;
}

bool setup_client_server() {
    client_server_mode = true;
    is_server = (shmem_my_pe() != 0);
    return is_server;
}

shmem_ml_py_cmd command_loop() {
    shmem_ml_cmd<uint8_t> *cmd = (shmem_ml_cmd<uint8_t>*)malloc(sizeof(*cmd));
    assert(cmd);
    size_t cmd_capacity = sizeof(*cmd);

    while (true) {
        int msg_len = mailbox_recv(cmd, cmd_capacity, &cmd_mailbox);
        if (msg_len > 0) {
            if (msg_len > cmd_capacity) {
                /*
                 * Not enough space in our cmd buffer to receive the current
                 * message, realloc and loop again.
                 */
                cmd = (shmem_ml_cmd<uint8_t>*)realloc(cmd, msg_len);
                assert(cmd);
                cmd_capacity = msg_len;
            } else {
                shmem_ml_py_cmd py_cmd = cmd->handler(cmd->cmd, &cmd->payload,
                        msg_len - offsetof(shmem_ml_cmd<uint8_t>, payload));
                if (py_cmd.get_cmd() != CMD_INVALID) {
                    free(cmd);
                    return py_cmd;
                }
            }
        }
    }

    abort();
}

void end_cmd() {
    inside_cmd--;
}

