#include "shmem_ml.hpp"
#include <stdio.h>

int main() {
    shmem_init();
    int pe = shmem_my_pe();
    int npes = shmem_n_pes();

    int M = 2000;
    int N = 5;
    {
        ShmemML2D mat(M, N);
        mat.clear(0.0);

        assert(mat.M() == M);
        assert(mat.N() == N);

        for (int i = 0; i < mat.M(); i++) {
            for (int j = 0; j < mat.N(); j++) {
                assert(mat.get(i, j) == 0.0);
            }
        }

        mat.sync();

        for (int i = 0; i < mat.M(); i++) {
            for (int j = 0; j < mat.N(); j++) {
                mat.set(i, j, i + j);
            }
        }

        mat.sync();

        for (int i = 0; i < mat.M(); i++) {
            for (int j = 0; j < mat.N(); j++) {
                assert(mat.get(i, j) == i + j);
            }
        }
    } 
    shmem_finalize();
    if (pe == 0) {
        printf("Success (%d PEs)\n", npes);
    }
    return 0;
}
