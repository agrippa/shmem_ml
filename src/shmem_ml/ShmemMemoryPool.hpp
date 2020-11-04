#ifndef _SHMEM_MEMORY_POOL_HPP
#define _SHMEM_MEMORY_POOL_HPP

#ifdef __cplusplus
extern "C" {
#endif
#include <dlmalloc/dlmalloc.h>
#include <shmem.h>
#ifdef __cplusplus
}
#endif

// #define TRACE_MEM

#include <assert.h>
#include <arrow/memory_pool.h>

class ShmemMemoryPool : public arrow::MemoryPool {
    public:
        explicit ShmemMemoryPool(int64_t pool_size) {
            _bytes_allocated = 0;
            _pool_size = pool_size;
            _pool = shmem_malloc(pool_size);
            assert(_pool);
            _allocator = shmemml_create_mspace_with_base(_pool, _pool_size, 0);
            assert(_allocator);
        }

        ~ShmemMemoryPool() override {
            shmemml_destroy_mspace(_allocator);
            shmem_free(_pool);
        }

        arrow::Status Allocate(int64_t size, uint8_t** out) override {
            arrow::Status status = arrow::Status::OK();
            if (size == 0) {
                *out = NULL;
            } else {
                void *ptr = shmemml_mspace_malloc(_allocator, size);
                if (ptr) {
                    _bytes_allocated += size;
                    *out = (uint8_t*)ptr;
                } else {
                    *out = NULL;
                    status = arrow::Status::OutOfMemory("ShmemMemoryPool::Allocate");
                }
            }
#ifdef TRACE_MEM
            fprintf(stderr, "PE %d Allocate size=%ld out=%p\n", shmem_my_pe(),
                    size, *out);
#endif
            return status;
        }

        arrow::Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override {
#ifdef TRACE_MEM
            fprintf(stderr, "PE %d Reallocate old_size=%ld new_size=%ld "
                    "*ptr=%p\n", shmem_my_pe(), old_size, new_size, *ptr);
#endif
            if (*ptr == NULL) {
                assert(old_size == 0);
                return Allocate(new_size, ptr);
            }

            if (new_size == 0) {
                Free(*ptr, old_size);
                *ptr = NULL;
                return arrow::Status::OK();
            }

            void *new_ptr = shmemml_mspace_realloc(_allocator, *ptr, new_size);
            if (new_ptr) {
                _bytes_allocated += (new_size - old_size);
                *ptr = (uint8_t*)new_ptr;
                return arrow::Status::OK();
            } else {
                return arrow::Status::OutOfMemory("ShmemMemoryPool::Reallocate");
            }
        }

        void Free(uint8_t* buffer, int64_t size) override {
#ifdef TRACE_MEM
            fprintf(stderr, "PE %d Free buffer=%p size=%ld\n", shmem_my_pe(),
                    buffer, size);
#endif
            if (buffer == NULL) {
                assert(size == 0);
            } else {
                _bytes_allocated -= size;
                /*
                 * Note that a common issue is a crash here when SHMEM-ML tries
                 * to free a stack-allocated object in main() after the
                 * programmer has called shmem_finalize. The simplest workaround
                 * is to add a local scope in main() that closes before
                 * shmem_finalize to ensure all symmetric objects are cleaned
                 * up.
                 */
                shmemml_mspace_free(_allocator, buffer);
            }
        }

        int64_t bytes_allocated() const override {
            return _bytes_allocated;
        }

        int64_t max_memory() const override {
            return _pool_size;
        }

        std::string backend_name() const override {
            return std::string("shmem");
        }

        static ShmemMemoryPool* get() {
            if (singleton == NULL) {
                int64_t pool_size = 1024 * 1024;
                if (getenv("SHMEM_ML_POOL_SIZE")) {
                    pool_size = atoi(getenv("SHMEM_ML_POOL_SIZE"));
                }
                singleton = new ShmemMemoryPool(pool_size);
            }
            return singleton;
        }

    private:
        int64_t _bytes_allocated;
        int64_t _pool_size;
        void* _pool;
        mspace _allocator;

        static ShmemMemoryPool *singleton;
};

#endif
