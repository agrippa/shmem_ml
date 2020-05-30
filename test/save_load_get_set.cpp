#include <shmem_ml.hpp>

int main(int argc, char **argv) {
    shmem_init();
    int pe = shmem_my_pe();

    {
        int64_t n = 12345;
        ShmemML1D<double> arr(n);

        int64_t start_edge_index = arr.local_slice_start();
        int64_t end_edge_index = arr.local_slice_end();

        for (int64_t i = start_edge_index; i < end_edge_index; i++) {
            arr.set(i, i);
        }

        arr.save("tmp.bin");

        ShmemML1D<double>* loaded = ShmemML1D<double>::load("tmp.bin");

        assert(arr.N() == loaded->N());
        assert(start_edge_index == loaded->local_slice_start());
        assert(end_edge_index == loaded->local_slice_end());

        for (int64_t i = 0; i < arr.N(); i++) {
            double a = arr.get(i);
            double b = loaded->get(i);
            assert(a == b);
        }

        delete loaded;
    }

    shmem_finalize();

    printf("PE %d success\n", pe);

    return 0;
}
