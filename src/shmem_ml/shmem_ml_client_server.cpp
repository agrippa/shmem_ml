#include <Python.h>
#include "shmem_ml.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <script.py>\n", argv[0]);
        return 1;
    }

    shmem_init();

    shmem_ml_client_server_launch([&] {
            Py_Initialize();
            FILE *fp = fopen(argv[1], "r");
            PyRun_SimpleFileExFlags(fp, argv[1], 1, NULL);
            Py_Finalize();
        });

    shmem_finalize();

    return 0;
}
