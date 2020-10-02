/*
 * This was taken from the dlmalloc.c to act as a standalone header
 * file for other parts of the OpenSHMEM library
 *
 * License as for dlmalloc.c
 */

#ifndef _DLMALLOC_H
#define _DLMALLOC_H 1

#include <sys/types.h>

#define MSPACES 1
#define ONLY_MSPACES 1
#define HAVE_MORECORE 0
#define DLMALLOC_HAVE_MMAP 0
#define DLMALLOC_HAVE_MREMAP 0
#define USE_LOCKS 0

typedef void *mspace;

extern mspace shmemml_create_mspace_with_base(void* base, size_t capacity, int locked);

extern size_t shmemml_destroy_mspace(mspace msp);

extern void* shmemml_mspace_malloc(mspace msp, size_t bytes);

extern void* shmemml_mspace_realloc(mspace msp, void* mem, size_t newsize);

extern void* mspace_memalign(mspace msp, size_t alignment, size_t bytes);

extern void shmemml_mspace_free(mspace msp, void *mem);

extern size_t mspace_footprint(mspace msp);

#endif /* _DLMALLOC_H */
