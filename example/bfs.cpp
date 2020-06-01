#include <shmem_ml.hpp>

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <generator/splittable_mrg.h>
#include <generator/make_graph.h>
#include <generator/graph_generator.h>
#include <generator/utils.h>

static int is_isolated = 0;

int64_t* compute_bfs_roots(int &num_bfs_roots, int64_t nglobalverts,
        ShmemML1D<int64_t>* verts, int64_t *neighbor_list_offsets) {
    int64_t* bfs_roots = (int64_t*)malloc(num_bfs_roots * sizeof(int64_t));
    assert(bfs_roots);

	uint64_t seed1 = 2, seed2 = 3;
    uint_fast32_t seed[5];
    make_mrg_seed(seed1, seed2, seed);

    uint64_t counter = 0;
    int bfs_root_idx;
    for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
        int64_t root;
        while (1) {
            double d[2];
            make_random_numbers(2, seed1, seed2, counter, d);
            root = (int64_t)((d[0] + d[1]) * nglobalverts) % nglobalverts;
            counter += 2;
            if (counter > 2 * nglobalverts) break;
            int is_duplicate = 0;
            int i;
            for (i = 0; i < bfs_root_idx; ++i) {
                if (root == bfs_roots[i]) {
                    is_duplicate = 1;
                    break;
                }
            }
            if (is_duplicate) continue; /* Everyone takes the same path here */

            int owning_pe = verts->owning_pe(root);
            if (owning_pe == shmem_my_pe()) {
                int64_t local_vert_offset = root - verts->local_slice_start();
                int64_t list_len = neighbor_list_offsets[local_vert_offset + 1] -
                    neighbor_list_offsets[local_vert_offset];
                is_isolated = (list_len == 0);

                for (int p = 0; p < shmem_n_pes(); p++) {
                    if (p == shmem_my_pe()) continue;
                    shmem_int_p(&is_isolated, is_isolated, p);
                }
            }

            shmem_barrier_all();

            if (is_isolated) continue;
        }
        bfs_roots[bfs_root_idx] = root;
    }
    num_bfs_roots = bfs_root_idx;
    return bfs_roots;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <output-edges-file> <# roots> <scale>\n",
                argv[0]);
        return 1;
    }

    shmem_init();
    int pe = shmem_my_pe();
    int npes = shmem_n_pes();

    {
        int SCALE = atoi(argv[3]);
        int64_t edgefactor = 16; // edges per vertex
        int64_t nvertices = (int64_t)(1) << SCALE;
        int64_t nedges = (int64_t)(edgefactor) << SCALE;
        ShmemML1D<packed_edge> edges(nedges);

        int64_t start_edge_index = edges.local_slice_start();
        int64_t end_edge_index = edges.local_slice_end();

        uint64_t seed1 = 2, seed2 = 3;
        uint_fast32_t seed[5];
        make_mrg_seed(seed1, seed2, seed);

        generate_kronecker_range(seed, SCALE, start_edge_index, end_edge_index,
                edges.raw_slice());

        ShmemML1D<int64_t> *verts = new ShmemML1D<int64_t>(nvertices, INT64_MAX);
        assert(verts);

        /*
         * From Graph500 spec:
         *   The graph generator creates a small number of multiple edges
         *   between two vertices as well as self-loops. Multiple edges,
         *   self-loops, and isolated vertices may be ignored in the subsequent
         *   kernels but must be included in the edge list provided to the first
         *   kernel.
         * We remove self loops here, and trim duplicate edges below.
         */

        ShmemML1D<long long> edges_per_pe(npes, (long long)0);
        for (int64_t i = 0; i < end_edge_index - start_edge_index; i++) {
            packed_edge curr = edges.get_local(i);
            int64_t v0 = get_v0_from_edge(&curr);
            int64_t v1 = get_v1_from_edge(&curr);

            if (v0 == v1) continue;

            int v0_pe = verts->owning_pe(v0);
            int v1_pe = verts->owning_pe(v1);
            edges_per_pe.atomic_add(v0_pe, 1);
            if (v1_pe != v0_pe) {
                edges_per_pe.atomic_add(v1_pe, 1);
            }
        }

        edges_per_pe.sync();

        long long max_edges_per_pe = edges_per_pe.max(-1);

        ShmemML1D<packed_edge> partitioned_edges(max_edges_per_pe * npes);
        ShmemML1D<int64_t> partitioned_edges_counter(npes, 0);
        for (int64_t i = 0; i < end_edge_index - start_edge_index; i++) {
            packed_edge curr = edges.get_local(i);
            int64_t v0 = get_v0_from_edge(&curr);
            int64_t v1 = get_v1_from_edge(&curr);
            if (v0 == v1) continue;

            int v0_pe = verts->owning_pe(v0);
            int v1_pe = verts->owning_pe(v1);

            long long index0 = partitioned_edges_counter.atomic_fetch_add(v0_pe, 1);
            partitioned_edges.set_local(v0_pe, index0, edges.get_local(i));

            if (v1_pe != v0_pe) {
                long long index1 = partitioned_edges_counter.atomic_fetch_add(v1_pe, 1);
                partitioned_edges.set_local(v1_pe, index1, edges.get_local(i));
            }
        }
        partitioned_edges.sync();

        int64_t n_local_verts = verts->local_slice_end() -
            verts->local_slice_start();
        int64_t* neighbors_per_vertex = (int64_t *)malloc(
                n_local_verts * sizeof(*neighbors_per_vertex));
        assert(neighbors_per_vertex);
        memset(neighbors_per_vertex, 0x00,
                n_local_verts * sizeof(*neighbors_per_vertex));
        for (int64_t i = 0; i < edges_per_pe.get_local(0); i++) {
            packed_edge curr = partitioned_edges.get_local(i);
            int64_t v0 = get_v0_from_edge(&curr);
            int64_t v1 = get_v1_from_edge(&curr);
            if (verts->owning_pe(v0) == pe) {
                neighbors_per_vertex[v0 - verts->local_slice_start()]++;
            }
            if (verts->owning_pe(v1) == pe) {
                neighbors_per_vertex[v1 - verts->local_slice_start()]++;
            }
        }
        int64_t neighbor_lists_len = 0;
        for (int i = 0; i < n_local_verts; i++) {
            neighbor_lists_len += neighbors_per_vertex[i];
        }

        int64_t *neighbor_list_offsets = (int64_t *)malloc(
                (n_local_verts + 1) * sizeof(*neighbor_list_offsets));
        assert(neighbor_list_offsets);
        neighbor_list_offsets[0] = 0;
        for (int i = 1; i <= n_local_verts; i++) {
            neighbor_list_offsets[i] = neighbor_list_offsets[i - 1] +
                neighbors_per_vertex[i - 1];
        }

        memset(neighbors_per_vertex, 0x00,
                n_local_verts * sizeof(*neighbors_per_vertex));

        // TODO delete duplicate edges
        int64_t* neighbor_lists = (int64_t *)malloc(
                neighbor_lists_len * sizeof(*neighbor_lists));
        assert(neighbor_lists);
        for (int64_t i = 0; i < edges_per_pe.get_local(0); i++) {
            packed_edge curr = partitioned_edges.get_local(i);
            int64_t v0 = get_v0_from_edge(&curr);
            int64_t v1 = get_v1_from_edge(&curr);
            if (verts->owning_pe(v0) == pe) {
                int64_t local_v0_offset = v0 - verts->local_slice_start();
                int64_t index = neighbor_list_offsets[local_v0_offset] +
                    neighbors_per_vertex[local_v0_offset];
                neighbor_lists[index] = v1;
                neighbors_per_vertex[local_v0_offset]++;
            }
            if (verts->owning_pe(v1) == pe) {
                int64_t local_v1_offset = v1 - verts->local_slice_start();
                int64_t index = neighbor_list_offsets[local_v1_offset] +
                    neighbors_per_vertex[local_v1_offset];
                neighbor_lists[index] = v0;
                neighbors_per_vertex[local_v1_offset]++;
            }
        }

        /*
         * Find and remove duplicates in neighbor lists. This is a pretty
         * inefficient way to do this at the moment.
         */
        int64_t count_duplicates = 0;
        for (int64_t i = 0; i < n_local_verts; i++) {
            int64_t *vert_neighbor_list = neighbor_lists +
                neighbor_list_offsets[i];
            int64_t vert_neighbor_list_len = neighbor_list_offsets[i + 1] -
                neighbor_list_offsets[i];
            for (int j = vert_neighbor_list_len - 1; j > 0; j--) {
                int duplicated = 0;
                for (int k = j - 1; k >= 0; k--) {
                    if (vert_neighbor_list[j] == vert_neighbor_list[k]) {
                        duplicated = 1;
                    }
                }

                if (duplicated) {
                    vert_neighbor_list[j] = vert_neighbor_list[vert_neighbor_list_len - 1];
                    vert_neighbor_list_len--;
                    count_duplicates++;
                }
            }
            int64_t old_vert_neighbor_list_len = neighbor_list_offsets[i + 1] -
                neighbor_list_offsets[i];
            int64_t delta = old_vert_neighbor_list_len - vert_neighbor_list_len;
            if (delta > 0) {
                /*
                 * Shift all neighbor lists and neighbor list offsets down by
                 * the difference
                 */
                int64_t next_offset = neighbor_list_offsets[i + 1];
                for (int64_t j = next_offset; j < neighbor_list_offsets[n_local_verts]; j++) {
                    neighbor_lists[j - delta] = neighbor_lists[j];
                }
                for (int64_t j = i + 1; j <= n_local_verts; j++) {
                    neighbor_list_offsets[j] -= delta;
                }
            }
        }
        // printf("PE %d deleted %ld duplicate edges out of %ld total local "
        //         "edges (%f%%)\n", pe, count_duplicates, neighbor_lists_len,
        //         100.0 * count_duplicates / neighbor_lists_len);

        /*
         * neighbor_lists stores each local vertex's neighbor list in a
         * contiguous 1D array. neighbor_list_offsets[V] stores the offset of
         * the neighbor list for local vertex V in neighbor_lists. The length of
         * local vertex V's neighbor list can be computed by
         * neighbor_list_offsets[V+1] - neighbor_list_offsets[V].
         */
        int num_bfs_roots = atoi(argv[2]);
        int64_t *bfs_roots = compute_bfs_roots(num_bfs_roots, nvertices, verts,
                neighbor_list_offsets);

        ShmemML1D<long long> changes(npes);

        for (int root_index = 0; root_index < num_bfs_roots; root_index++) {
            unsigned long long start_time_us = shmem_ml_current_time_us();

            int64_t root = bfs_roots[root_index];
            int root_pe = verts->owning_pe(root);
            if (root_pe == pe) {
                verts->set(root, -(root + 1));
            }

            int any_changes;
            int iter = 0;
            do {
                unsigned nchanges = 0;
                for (int64_t i = 0; i < n_local_verts; i++) {
                    if (verts->get_local(i) < 0) {
                        int64_t parent = (-verts->get_local(i)) - 1;
                        int64_t *vert_neighbor_list = neighbor_lists +
                            neighbor_list_offsets[i];
                        int64_t vert_neighbor_list_len =
                            neighbor_list_offsets[i + 1] -
                            neighbor_list_offsets[i];
                        for (int j = 0; j < vert_neighbor_list_len; j++) {
                            int64_t found_val = verts->atomic_cas(
                                    vert_neighbor_list[j], INT64_MAX,
                                    -(verts->local_slice_start() + i + 1));
                            nchanges += (found_val == INT64_MAX);
                        }
                        verts->set_local(i, parent);
                    }
                }
                changes.set_local(0, nchanges);
                long long n_global_changes = changes.sum(0);
                if (pe == 0) {
                    printf("Root %d (%ld) iter %d changes %ld\n", root_index,
                            root, iter, n_global_changes);
                }
                any_changes = (n_global_changes > 0);
                iter++;
            } while(any_changes);

            unsigned long long elapsed_time_us = shmem_ml_current_time_us() -
                start_time_us;
            if (pe == 0) {
                printf("Completed root %d in %f s\n", root_index,
                        elapsed_time_us / 1000000.0);
            }

            delete verts;
            ShmemML1D<int64_t> *verts = new ShmemML1D<int64_t>(nvertices,
                    INT64_MAX);
            assert(verts);
        }

        edges.save(argv[1]);

        delete verts;
    }

    shmem_finalize();

    return 0;
}
