#include <Python.h>
#include "shmem_ml.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <script.py>\n", argv[0]);
        return 1;
    }

    shmem_init();

    py_shmem_ml_client_server_launch(argv[1]);

    shmem_finalize();

    return 0;
}
