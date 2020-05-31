#include <time.h>
#include <sys/time.h>

#include "ShmemMemoryPool.hpp"
#include "shmem_ml.hpp"

unsigned long long shmem_ml_current_time_us() {
    struct timespec monotime;
    clock_gettime(CLOCK_MONOTONIC, &monotime);
    return monotime.tv_sec * 1000000ULL + monotime.tv_nsec / 1000;
}
