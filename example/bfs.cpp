#include <shmem_ml.hpp>

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <generator/splittable_mrg.h>
#include <generator/graph_generator.h>
#include <generator/utils.h>


int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output-edges-file>\n", argv[0]);
        return 1;
    }

    shmem_init();
    int pe = shmem_my_pe();

    {
        int64_t nedges = 12345;
        ShmemML1D<packed_edge> edges(nedges);

        int64_t start_edge_index = edges.local_slice_start();
        int64_t end_edge_index = edges.local_slice_end();

        int SCALE = 16;
        uint64_t seed1 = 2, seed2 = 3;
        uint_fast32_t seed[5];
        make_mrg_seed(seed1, seed2, seed);

        generate_kronecker_range(seed, SCALE, start_edge_index, end_edge_index,
                edges.raw_slice());

        edges.save(argv[1]);

        ShmemML1D<packed_edge>* loaded = ShmemML1D<packed_edge>::load(argv[1]);

        assert(edges.N() == loaded->N());
        assert(start_edge_index == loaded->local_slice_start());
        assert(end_edge_index == loaded->local_slice_end());

        for (int64_t i = 0; i < edges.N(); i++) {
            packed_edge a = edges.get(i);
            packed_edge b = loaded->get(i);
            assert(get_v0_from_edge(&a) == get_v0_from_edge(&b));
            assert(get_v1_from_edge(&a) == get_v1_from_edge(&b));
        }

        delete loaded;
    }

    shmem_finalize();

    printf("PE %d success\n", pe);

    return 0;
}
