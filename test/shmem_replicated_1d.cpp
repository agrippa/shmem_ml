#include "shmem_ml.hpp"
#include <stdio.h>

int main() {
    shmem_init();
    int pe = shmem_my_pe();
    int npes = shmem_n_pes();

    {
        // Test N(), local_slice_start(), local_slice_end(), owning_pe()
        int per_pe = 10;
        ReplicatedShmemML1D<double> arr(per_pe);
        assert(arr.N() == per_pe);
        assert(arr.local_slice_start() == 0);
        assert(arr.local_slice_end() == per_pe);
        for (int i = 0; i < per_pe; i++) {
            assert(arr.owning_pe(i) == pe);
        }
    }

    {
        // Set set/get
        ReplicatedShmemML1D<double> arr(1000);
        for (int i = 0; i < 1000; i++) {
            arr.set(i, i);
        }
        for (int i = 0; i < 1000; i++) {
            assert(arr.get(i) == i);
        }
    }

    {
        // Test apply_ip
        ReplicatedShmemML1D<double> arr(1000);
        for (int i = 0; i < 1000; i++) {
            arr.set(i, i);
        }
        arr.apply_ip([&] (int64_t gind, int64_t lind, double& v) {
                    v = 2 * v;
                });
        for (int i = 0; i < 1000; i++) {
            assert(arr.get(i) == 2 * i);
        }
    }

    {
        // Test atomics
        ReplicatedShmemML1D<double> d_arr(3000);
        d_arr.clear(0.0);

        // Atomic add
        d_arr.atomic_add(2000, 4.0);

        // Atomic fetch add
        double d = d_arr.atomic_fetch_add(1000, 5.0);
        assert(d == 0);

        // Atomic CAS
        d = d_arr.atomic_cas(500, 0.0, 12.0);
        assert(d == 0.0);

        d_arr.sync();

        // Atomic add
        assert(d_arr.get(2000) == 4.0);

        // Atomic fetch add
        assert(d_arr.get(1000) == 5.0);

        // Atomic CAS
        assert(d_arr.get(500) == 12.0);
    }

    {
        // Test reductions
        ReplicatedShmemML1D<double> d_arr(3000);
        d_arr.clear(0.0);

        d_arr.apply_ip([&] (int64_t global_i, int64_t local_i, double &v) {
                    v = global_i;
                });

        double d_max = d_arr.max(0.0);

        assert(d_max == (3000 - 1));

        double d_sum = d_arr.sum(0.0);

        assert(d_sum == ((3000-1) * (3000-1+1)) / 2);
    }

    {
        // Test combines on replicated arrays
        ReplicatedShmemML1D<unsigned> d_arr(3000);
        d_arr.clear(0.0);

        if (pe == 0) {
            d_arr.apply_ip([&] (int64_t global_i, int64_t local_i, unsigned &v) {
                        v = global_i;
                    });
        }

        d_arr.bcast(0);

        for (int i = 0; i < d_arr.N(); i++) {
            assert(d_arr.get(i) == i);
        }

        d_arr.reduce_all_sum();

        for (int i = 0; i < d_arr.N(); i++) {
            assert(d_arr.get(i) == npes * i);
        }

        d_arr.reduce_all_or();

        for (int i = 0; i < d_arr.N(); i++) {
            assert(d_arr.get(i) == npes * i);
        }
    }
    
    shmem_finalize();
    if (pe == 0) {
        printf("Success (%d PEs)\n", npes);
    }
    return 0;
}
