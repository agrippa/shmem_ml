#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <shmem.h>

#include "mailbox_buffer.hpp"

void mailbox_buffer_init(mailbox_buffer_t *buf, mailbox_t *mbox,
        int npes, size_t msg_size, size_t buffer_size_per_pe) {
    buf->mbox = mbox;
    buf->npes = npes;
    buf->msg_size = msg_size;
    buf->buffer_size_per_pe = buffer_size_per_pe;

    buf->nbuffered_per_pe = (unsigned *)malloc(npes * sizeof(unsigned));
    assert(buf->nbuffered_per_pe);
    memset(buf->nbuffered_per_pe, 0x00, npes * sizeof(unsigned));

    buf->buffers = (mailbox_msg_header_t**)malloc(npes * sizeof(buf->buffers[0]));
    assert(buf->buffers);
    for (int i = 0; i < npes; i++) {
        buf->buffers[i] = mailbox_allocate_msg(buffer_size_per_pe * msg_size);
    }
}

int mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, mailbox_buffer_t *buf) {
    assert(msg_len == buf->msg_size);

    unsigned nbuffered = buf->nbuffered_per_pe[target_pe];
    assert(nbuffered <= buf->buffer_size_per_pe);

    mailbox_msg_header_t *pe_buf = buf->buffers[target_pe];

    if (nbuffered == buf->buffer_size_per_pe) {
        // flush
        int success = mailbox_send(pe_buf, nbuffered * buf->msg_size,
                target_pe, max_tries, buf->mbox);
        if (success) {
            buf->nbuffered_per_pe[target_pe] = 0;
        } else {
            return 0;
        }
    }

    nbuffered = buf->nbuffered_per_pe[target_pe];
    char *dst = ((char *)(pe_buf + 1)) + (nbuffered * buf->msg_size);
    memcpy(dst, msg, msg_len);
    buf->nbuffered_per_pe[target_pe] = nbuffered + 1;
    return 1;
}

int mailbox_buffer_flush(mailbox_buffer_t *buf, int max_tries) {
    for (int p = 0; p < buf->npes; p++) {
        const unsigned nbuffered = buf->nbuffered_per_pe[p];
        assert(nbuffered <= buf->buffer_size_per_pe);

        mailbox_msg_header_t *pe_buf = buf->buffers[p];
        if (nbuffered > 0) {
            unsigned count_loops = 0;
            int printed_warning = 0;

            int success = mailbox_send(pe_buf, nbuffered * buf->msg_size,
                    p, max_tries, buf->mbox);
            if (success) {
                buf->nbuffered_per_pe[p] = 0;
            } else {
                return 0;
            }
        }
    }
    return 1;
}
