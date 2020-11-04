#include "shmem_ml.hpp"
#include <stdio.h>

int main() {
    shmem_init();
    int pe = shmem_my_pe();
    int npes = shmem_n_pes();

    {
        // Test N(), local_slice_start(), local_slice_end(), owning_pe()
        int per_pe = 10;
        ShmemML1D<int64_t> i_arr(npes * per_pe);
        assert(i_arr.N() == npes * per_pe);
        assert(i_arr.local_slice_start() == pe * per_pe);
        assert(i_arr.local_slice_end() == (pe + 1) * per_pe);
        for (int i = 0; i < npes * per_pe; i++) {
            int expected = i / per_pe;
            assert(i_arr.owning_pe(i) == expected);
        }
    }

    {
        // Test clear() on several data types
        ShmemML1D<double> d_arr(3000);
        ShmemML1D<uint32_t> u_arr(4000);
        ShmemML1D<int64_t> i_arr(5000);
        d_arr.clear(3.0);
        u_arr.clear(42);
        i_arr.clear(17);

        for (int64_t i = d_arr.local_slice_start(); i < d_arr.local_slice_end();
                i++) {
            assert(d_arr.get(i) == 3.0);
        }
        for (int64_t i = u_arr.local_slice_start(); i < u_arr.local_slice_end();
                i++) {
            assert(u_arr.get(i) == 42);
        }
        for (int64_t i = i_arr.local_slice_start(); i < i_arr.local_slice_end();
                i++) {
            assert(i_arr.get(i) == 17);
        }
    }

    {
        // Test set/get
        ShmemML1D<double> d_arr(3000);
        ShmemML1D<uint32_t> u_arr(4000);
        ShmemML1D<int64_t> i_arr(5000);
        if (pe == 0) {
            for (int i = 0; i < d_arr.N(); i++) {
                d_arr.set(i, i);
            }
            for (int i = 0; i < u_arr.N(); i++) {
                u_arr.set(i, 2 * i);
            }
            for (int i = 0; i < i_arr.N(); i++) {
                i_arr.set(i, 3 * i);
            }
        }
        d_arr.sync();
        u_arr.sync();
        i_arr.sync();
        if (pe == npes - 1) {
            for (int i = 0; i < d_arr.N(); i++) {
                assert(d_arr.get(i) == i);
            }
            for (int i = 0; i < u_arr.N(); i++) {
                assert(u_arr.get(i) == 2 * i);
            }
            for (int i = 0; i < i_arr.N(); i++) {
                assert(i_arr.get(i) == 3 * i);
            }
        }
    }

    {
        // Test apply_ip
        ShmemML1D<double> d_arr(3000);
        ShmemML1D<uint32_t> u_arr(4000);
        ShmemML1D<int64_t> i_arr(5000);
        d_arr.clear(0.0);
        u_arr.clear(0);
        i_arr.clear(0);

        d_arr.apply_ip([&] (int64_t global_i, int64_t local_i, double &v) {
                    v = 2 * global_i;
                });
        u_arr.apply_ip([&] (int64_t global_i, int64_t local_i, uint32_t& v) {
                    v = 2 * global_i;
                });
        i_arr.apply_ip([&] (int64_t global_i, int64_t local_i, int64_t& v) {
                    v = 2 * global_i;
                });
        d_arr.sync();
        u_arr.sync();
        i_arr.sync();

        for (int64_t i = d_arr.local_slice_start(); i < d_arr.local_slice_end();
                i++) {
            assert(d_arr.get(i) == 2 * i);
        }
        for (int64_t i = u_arr.local_slice_start(); i < u_arr.local_slice_end();
                i++) {
            assert(u_arr.get(i) == 2 * i);
        }
        for (int64_t i = i_arr.local_slice_start(); i < i_arr.local_slice_end();
                i++) {
            assert(i_arr.get(i) == 2 * i);
        }

        ShmemML1DIndex index(3000);
        for (int64_t i = d_arr.local_slice_start(); i < d_arr.local_slice_end();
                i++) {
            if (i % 2 == 0) {
                index.add(i);
            }
        }
        d_arr.apply_ip(index,
                [&] (int64_t global_i, int64_t local_i, double &v) {
                    v = 2 * v;
                });
        d_arr.sync();
        for (int64_t i = d_arr.local_slice_start(); i < d_arr.local_slice_end();
                i++) {
            if (i % 2 == 0) {
                assert(d_arr.get(i) == 4 * i);
            } else {
                assert(d_arr.get(i) == 2 * i);
            }
        }
    }

    {
        // Test atomics
        ShmemML1D<double> d_arr(3000);
        ShmemML1D<uint32_t> u_arr(4000);
        ShmemML1D<int64_t> i_arr(5000);
        d_arr.clear(0.0);
        u_arr.clear(0);
        i_arr.clear(0);

        if (pe == 0) {
            // Atomic add
            d_arr.atomic_add(2000, 4.0);
            u_arr.atomic_add(2000, 4);
            i_arr.atomic_add(2000, 4);

            // Atomic fetch add
            double d = d_arr.atomic_fetch_add(1000, 5.0);
            assert(d == 0);
            uint32_t u = u_arr.atomic_fetch_add(1000, 5);
            assert(u == 0);
            int64_t i = i_arr.atomic_fetch_add(1000, 5);
            assert(i == 0);

            // Atomic CAS
            d = d_arr.atomic_cas(500, 0.0, 12.0);
            assert(d == 0.0);
            u = u_arr.atomic_cas(500, 0, 12);
            assert(u == 0);
            i = i_arr.atomic_cas(500, 0, 12);
            assert(i == 0);

            // Atomic OR
            u_arr.atomic_or(200, 0x1);
            i_arr.atomic_or(200, 0x1);
        }

        d_arr.sync();
        u_arr.sync();
        i_arr.sync();

        if (pe == npes - 1) {
            // Atomic add
            assert(d_arr.get(2000) == 4.0);
            assert(u_arr.get(2000) == 4);
            assert(i_arr.get(2000) == 4);

            // Atomic fetch add
            assert(d_arr.get(1000) == 5.0);
            assert(u_arr.get(1000) == 5);
            assert(i_arr.get(1000) == 5);

            // Atomic CAS
            assert(d_arr.get(500) == 12.0);
            assert(u_arr.get(500) == 12);
            assert(i_arr.get(500) == 12);

            // Atomic OR
            assert(u_arr.get(200) == 1);
            assert(i_arr.get(200) == 1);
        }
    }

    {
        // Test reductions
        ShmemML1D<double> d_arr(3000);
        ShmemML1D<uint32_t> u_arr(4000);
        ShmemML1D<int64_t> i_arr(5000);
        d_arr.clear(0.0);
        u_arr.clear(0);
        i_arr.clear(0);

        d_arr.apply_ip([&] (int64_t global_i, int64_t local_i, double &v) {
                    v = global_i;
                });
        u_arr.apply_ip([&] (int64_t global_i, int64_t local_i, uint32_t& v) {
                    v = global_i;
                });
        i_arr.apply_ip([&] (int64_t global_i, int64_t local_i, int64_t& v) {
                    v = global_i;
                });
        d_arr.sync();
        u_arr.sync();
        i_arr.sync();

        double d_max = d_arr.max(0.0);
        uint32_t u_max = u_arr.max(0);
        int64_t i_max = i_arr.max(0);

        assert(d_max == (3000 - 1));
        assert(u_max == (4000 - 1));
        assert(i_max == (5000 - 1));

        double d_sum = d_arr.sum(0.0);
        uint32_t u_sum = u_arr.sum(0);
        int64_t i_sum = i_arr.sum(0);

        assert(d_sum == ((3000-1) * (3000-1+1)) / 2);
        assert(u_sum == ((4000-1) * (4000-1+1)) / 2);
        assert(i_sum == ((5000-1) * (5000-1+1)) / 2);
    }
    
    shmem_finalize();
    if (pe == 0) {
        printf("Success (%d PEs)\n", npes);
    }
    return 0;
}
