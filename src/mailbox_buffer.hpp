#ifndef _MAILBOX_BUFFER_H
#define _MAILBOX_BUFFER_H

#include "mailbox.hpp"

typedef struct _mailbox_buffer_t {
    mailbox_t *mbox;
    int npes;
    size_t msg_size;
    size_t buffer_size_per_pe;

    unsigned *nbuffered_per_pe;

    mailbox_msg_header_t** buffers;
} mailbox_buffer_t;

void mailbox_buffer_init(mailbox_buffer_t *buf, mailbox_t *mbox,
        int npes, size_t msg_size, size_t buffer_size_per_pe);

int mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, mailbox_buffer_t *buf);

int mailbox_buffer_flush(mailbox_buffer_t *buf, int max_tries);

#endif // _HVR_MAILBOX_BUFFER_H
