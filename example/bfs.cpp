#include <shmem_ml.hpp>

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <generator/splittable_mrg.h>
extern "C" {
#include <generator/make_graph.h>
}
#include <generator/graph_generator.h>
#include <generator/utils.h>

#include <cstdlib>
#include <pthread.h>
#include <unistd.h>

#ifdef CRAYPAT
#include <pat_api.h>
#endif

// #define SHARED_VISITED

static int *is_isolated = NULL;
static long *psync = NULL;

#define BITS_PER_BYTE 8
#define BITS_PER_INT (sizeof(unsigned) * BITS_PER_BYTE)

class bitvector {
    public:
        bitvector(int64_t nglobalverts) {
            const size_t visited_ints = ((nglobalverts + BITS_PER_INT - 1) /
                    BITS_PER_INT);
            visited = new ReplicatedShmemML1D<unsigned>(visited_ints);
            visited->zero();
        }

        void clear() {
            visited->zero();
        }

        void sync() {
            visited->reduce_all_or();
        }

        inline int is_set(const uint64_t global_vertex_id) {
            const unsigned word_index = global_vertex_id / BITS_PER_INT;
            const int bit_index = global_vertex_id - (word_index * BITS_PER_INT);
            // const int bit_index = global_vertex_id % BITS_PER_INT;
            const unsigned mask = ((unsigned)1 << bit_index);

            return (((visited->get(word_index) & mask) > 0) ? 1 : 0);
        }

        inline void set(const uint64_t global_vertex_id) {
            const int word_index = global_vertex_id / BITS_PER_INT;
            const int bit_index = global_vertex_id - (word_index * BITS_PER_INT);
            // const int bit_index = global_vertex_id % BITS_PER_INT;
            const unsigned mask = ((unsigned)1 << bit_index);

            visited->atomic_or(word_index, mask);
        }

    private:
        ReplicatedShmemML1D<unsigned>* visited;
};


int compar(const void* a,const void* b) {
    const int64_t pa = *((const int64_t *)a);
    const int64_t pb = *((const int64_t *)b);
    return (pa > pb) ? 1 : ((pa < pb) ? -1 : 0);
}

int64_t* compute_bfs_roots(int &num_bfs_roots, int64_t nglobalverts,
        ShmemML1D<int64_t>* verts, int64_t *neighbor_list_offsets) {
    int64_t* bfs_roots = (int64_t*)malloc(num_bfs_roots * sizeof(int64_t));
    assert(bfs_roots);

	uint64_t seed1 = 2, seed2 = 3;
    uint_fast32_t seed[5];
    make_mrg_seed(seed1, seed2, seed);

    uint64_t counter = 0;
    int bfs_root_idx;
    unsigned ntries = 0;
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
                *is_isolated = (list_len == 0);
            }

            shmem_barrier_all();
            shmem_broadcast32(is_isolated, is_isolated, 1, owning_pe, 0, 0, shmem_n_pes(), psync);
            shmem_barrier_all();

            int local_is_isolated = *is_isolated;

            if (!local_is_isolated) break;

            ntries++;
        }
        bfs_roots[bfs_root_idx] = root;
    }
    num_bfs_roots = bfs_root_idx;
    if (shmem_my_pe() == 0) {
        fprintf(stderr, "Took %u tries to generate %d roots\n", ntries,
                num_bfs_roots);
    }
    return bfs_roots;
}

static void *aborting_thread(void *user_data) {
    int nseconds = 600;

    const unsigned long long start = shmem_ml_current_time_us();
    while (shmem_ml_current_time_us() - start < nseconds * 1000000) {
        sleep(10);
    }

    fprintf(stderr, "BFS forcibly aborting\n");
    abort(); // Get a core dump

    return NULL;
}

static unsigned nchanges;

#ifdef ATOMICS_AS_MSGS
ShmemML1DIndex *changed_buf_0 = NULL;
ShmemML1DIndex *changed_buf_1 = NULL;
ShmemML1DIndex *curr_changed = NULL;
ShmemML1DIndex *next_changed = NULL;
void verts_atomics_cb(ShmemML1D<int64_t>* verts, int64_t global_index, atomics_msg_op_t op,
        int64_t prev_val, int64_t new_val) {
    if (op == CAS) {
        if (prev_val == INT64_MAX) {
            next_changed->add(global_index);
            nchanges++;
        }
    }
}
#endif

int main(int argc, char **argv) {

#ifdef CRAYPAT
            PAT_record(PAT_STATE_OFF);
#endif

    if (argc != 4) {
        fprintf(stderr, "usage: %s <output-edges-file> <# roots> <scale>\n",
                argv[0]);
        return 1;
    }

#if 0
    int tl;
    int err = shmem_init_thread(SHMEM_THREAD_SINGLE, &tl);
    assert(err == 0);
    assert(tl == SHMEM_THREAD_SINGLE);
    // if (tl != SHMEM_THREAD_SINGLE) {
    //     fprintf(stderr, "Error: requested single but got %s (%d)\n",
    //             tl == SHMEM_THREAD_FUNNELED ? "funneled" :
    //             (tl == SHMEM_THREAD_SERIALIZED ? "serialized" :
    //              (tl == SHMEM_THREAD_MULTIPLE ? "multiple" : "unknown")), tl);
    //     exit(1);
    // }
#else
    shmem_init();
#endif

    int pe = shmem_my_pe();
    int npes = shmem_n_pes();
    is_isolated = (int*)shmem_malloc(sizeof(*is_isolated));
    assert(is_isolated);
    psync = (long *)shmem_malloc(SHMEM_BCAST_SYNC_SIZE * sizeof(*psync));
    assert(psync);
    for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }

#if 0
    pthread_t aborting_pthread;
    const int pthread_err = pthread_create(&aborting_pthread, NULL,
            aborting_thread, NULL);
    assert(pthread_err == 0);
#endif

    {
        unsigned long long start_setup = shmem_ml_current_time_us();

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

        changed_buf_0 = new ShmemML1DIndex(nvertices);
        changed_buf_1 = new ShmemML1DIndex(nvertices);

        generate_kronecker_range(seed, SCALE, start_edge_index, end_edge_index,
                edges.raw_slice());

#ifdef ATOMICS_AS_MSGS
        ShmemML1D<int64_t> *verts = new ShmemML1D<int64_t>(nvertices, 1, verts_atomics_cb);
#else
        ShmemML1D<int64_t> *verts = new ShmemML1D<int64_t>(nvertices);
#endif
        assert(verts);
        verts->clear(INT64_MAX);

        int64_t n_local_verts = verts->local_slice_end() -
            verts->local_slice_start();
        if (pe == 0) {
            fprintf(stderr, "Running with scale=%d # global verts=%ld # global "
                    "edges=%ld # verts per PE=%ld # PEs=%d\n", SCALE, nvertices, nedges,
                    n_local_verts, npes);
        }

        /*
         * From Graph500 spec:
         *   The graph generator creates a small number of multiple edges
         *   between two vertices as well as self-loops. Multiple edges,
         *   self-loops, and isolated vertices may be ignored in the subsequent
         *   kernels but must be included in the edge list provided to the first
         *   kernel.
         * We remove self loops here, and trim duplicate edges below.
         */

        ShmemML1D<long long> edges_per_pe(npes);
        edges_per_pe.clear(0);

        edges.apply_ip([verts, &edges_per_pe] (int64_t global_index, int64_t local_index, packed_edge curr) {
            int64_t v0 = get_v0_from_edge(&curr);
            int64_t v1 = get_v1_from_edge(&curr);

            if (v0 == v1) return;

            int v0_pe = verts->owning_pe(v0);
            int v1_pe = verts->owning_pe(v1);
            edges_per_pe.atomic_add(v0_pe, 1);
            if (v1_pe != v0_pe) {
                edges_per_pe.atomic_add(v1_pe, 1);
            }
        });

        edges_per_pe.sync();

        long long max_edges_per_pe = edges_per_pe.max(-1);

        ShmemML1D<packed_edge> partitioned_edges(max_edges_per_pe * npes);
        ShmemML1D<int64_t> partitioned_edges_counter((int64_t)npes);
        partitioned_edges_counter.clear(0);

        edges.apply_ip([verts, &partitioned_edges_counter, &partitioned_edges, &edges] (int64_t global_index, int64_t local_index, packed_edge curr) {
            int64_t v0 = get_v0_from_edge(&curr);
            int64_t v1 = get_v1_from_edge(&curr);
            if (v0 == v1) return;

            int v0_pe = verts->owning_pe(v0);
            int v1_pe = verts->owning_pe(v1);

            long long index0 = partitioned_edges_counter.atomic_fetch_add(v0_pe, 1);
            partitioned_edges.set_local(v0_pe, index0, curr);

            if (v1_pe != v0_pe) {
                long long index1 = partitioned_edges_counter.atomic_fetch_add(v1_pe, 1);
                partitioned_edges.set_local(v1_pe, index1, curr);
            }
        });
        partitioned_edges.sync();
        unsigned long long done_partitioning = shmem_ml_current_time_us();
        if (pe == 0) {
            fprintf(stderr, "Partitioning edges across PEs took %f s\n",
                    (done_partitioning - start_setup) / 1000000.0);
        }

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
        shmem_barrier_all();
        unsigned long long done_neighbors_lists = shmem_ml_current_time_us();
        if (pe == 0) {
            fprintf(stderr, "Constructing neighbors lists took %f s\n",
                    (done_neighbors_lists - done_partitioning) / 1000000.0);
        }

        /*
         * Find and remove duplicates in neighbor lists. This is a pretty
         * inefficient way to do this at the moment.
         */
        int64_t count_duplicates = 0;
        int64_t index = 0;
        for (int64_t i = 0; i < n_local_verts; i++) {
            int64_t *vert_neighbor_list = neighbor_lists +
                neighbor_list_offsets[i];
            int64_t vert_neighbor_list_len = neighbor_list_offsets[i + 1] -
                neighbor_list_offsets[i];


            int64_t new_neighbor_list_offset = index;
            if (vert_neighbor_list_len > 0) {
                qsort(vert_neighbor_list, vert_neighbor_list_len,
                        sizeof(*vert_neighbor_list), compar);

                neighbor_lists[index++] = vert_neighbor_list[0];
                for (int i = 1; i < vert_neighbor_list_len; i++) {
                    if (vert_neighbor_list[i] != neighbor_lists[index - 1]) {
                        neighbor_lists[index++] = vert_neighbor_list[i];
                    }
                }
            }

            neighbor_list_offsets[i] = new_neighbor_list_offset;
        }
        neighbor_list_offsets[n_local_verts] = index;

        shmem_barrier_all();
        unsigned long long done_deduplicating = shmem_ml_current_time_us();
        if (pe == 0) {
            fprintf(stderr, "Deduplicating took %f s\n",
                    (done_deduplicating - done_neighbors_lists) / 1000000.0);
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

        unsigned long long done_roots = shmem_ml_current_time_us();

        if (pe == 0) {
            fprintf(stderr, "Generating roots took %f s\n",
                    (done_roots - done_deduplicating) / 1000000.0);
            fprintf(stderr, "Setup took %f s, scale=%d # roots=%d\n",
                    (done_roots - start_setup) / 1000000.0, SCALE, num_bfs_roots);
        }

        bitvector *visited = new bitvector(nvertices);
        assert(visited);

        for (int root_index = 0; root_index < num_bfs_roots; root_index++) {
            curr_changed = changed_buf_0;
            next_changed = changed_buf_1;
            curr_changed->clear();
            next_changed->clear();

            int64_t root = bfs_roots[root_index];
            int root_pe = verts->owning_pe(root);
            if (root_pe == pe) {
                verts->set(root, -(root + 1));
                curr_changed->add(root);
            }
            visited->set(root);

#ifdef CRAYPAT
            PAT_record(PAT_STATE_ON);
#endif

            shmem_barrier_all();
            unsigned long long start_time_us = shmem_ml_current_time_us();
            int any_changes;
            int iter = 0;
            do {
                nchanges = 0;
                unsigned nattempts = 0;
                verts->apply_ip(curr_changed,
                    [&verts, &neighbor_lists, &neighbor_list_offsets, &visited, &nattempts] (
                        const int64_t global_index,
                        const int64_t local_index,
                        int64_t vert_parent) {
                    assert(vert_parent < 0);
                    int64_t parent = (-vert_parent) - 1;
    
                    int64_t *vert_neighbor_list = neighbor_lists +
                        neighbor_list_offsets[local_index];
                    int64_t vert_neighbor_list_len =
                        neighbor_list_offsets[local_index + 1] -
                        neighbor_list_offsets[local_index];

                    for (int j = 0; j < vert_neighbor_list_len; j++) {
                        int64_t neighbor = vert_neighbor_list[j];
                        if (!visited->is_set(neighbor)) {
#ifdef ATOMICS_AS_MSGS
                            verts->atomic_cas_msg(neighbor, INT64_MAX,
                                    -(global_index + 1));
#else
                            int64_t found_val = verts->atomic_cas(
                                    neighbor, INT64_MAX,
                                    -(global_index + 1));
                            if (found_val == INT64_MAX) {
                                nchanges++;
                            }
#endif
                            visited->set(neighbor);
                        }
                    }
                    int64_t old = verts->atomic_cas(global_index,
                            vert_parent, parent);
                    assert(old == vert_parent);
                    visited->set(global_index);
                });

                verts->sync();

                ShmemML1DIndex *tmp = curr_changed;
                curr_changed = next_changed;
                next_changed = tmp;
                next_changed->clear();

                changes.set_local(0, nchanges);
                long long n_global_changes = changes.sum(0);
                if (pe == 0) {
                    printf("Root %d (%ld) iter %d changes %ld (local changes "
                            "%u)\n",
                            root_index, root, iter, n_global_changes, nchanges);
                }
                any_changes = (n_global_changes > 0);
                iter++;

#ifdef SHARED_VISITED
                if (any_changes) {
                    visited->sync();
                }
#endif
            } while(any_changes);

#ifdef CRAYPAT
            PAT_record(PAT_STATE_OFF);
#endif

            shmem_barrier_all();
            unsigned long long elapsed_time_us = shmem_ml_current_time_us() -
                start_time_us;
            if (pe == 0) {
                printf("Completed root %d in %f s\n", root_index,
                        elapsed_time_us / 1000000.0);
            }

            delete verts;
#ifdef ATOMICS_AS_MSGS
            verts = new ShmemML1D<int64_t>(nvertices, 1, verts_atomics_cb);
#else
            verts = new ShmemML1D<int64_t>(nvertices);
#endif
            assert(verts);
            verts->clear(INT64_MAX);

            visited->clear();
        }

        edges.save(argv[1]);

        delete verts;
    }

    shmem_finalize();

    return 0;
}
