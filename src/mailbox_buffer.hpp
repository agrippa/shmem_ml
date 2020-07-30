#ifndef _MAILBOX_BUFFER_H
#define _MAILBOX_BUFFER_H

#include "mailbox.hpp"

typedef struct _mailbox_header_wrapper_t {
    int pe;
    struct _mailbox_header_wrapper_t *next;
    struct _mailbox_header_wrapper_t *prev;

    struct _mailbox_msg_header_t *msg;
    shmem_ctx_t ctx;
} mailbox_header_wrapper_t;

typedef struct _mailbox_buffer_t {
    mailbox_t *mbox;
    int npes;
    size_t msg_size;
    size_t buffer_size_per_pe;

    unsigned *nbuffered_per_pe;

    mailbox_header_wrapper_t **active_buffers;
    mailbox_header_wrapper_t *free_pool;
    mailbox_header_wrapper_t *pending_pool_head;
    mailbox_header_wrapper_t *pending_pool_tail;
} mailbox_buffer_t;

void mailbox_buffer_init(mailbox_buffer_t *buf, mailbox_t *mbox,
        int npes, size_t msg_size, size_t buffer_size_per_pe,
        size_t buffer_pool_size);

int mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, mailbox_buffer_t *buf);

int mailbox_buffer_flush(mailbox_buffer_t *buf, int max_tries);

#endif // _HVR_MAILBOX_BUFFER_H
