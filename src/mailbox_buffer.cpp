#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <shmem.h>

#include "mailbox_buffer.hpp"

void mailbox_buffer_init(mailbox_buffer_t *buf, mailbox_t *mbox,
        int npes, size_t msg_size, size_t buffer_size_per_pe,
        size_t buffer_pool_size) {
    assert(buffer_pool_size >= npes);
    buf->mbox = mbox;
    buf->npes = npes;
    buf->msg_size = msg_size;
    buf->buffer_size_per_pe = buffer_size_per_pe;

    buf->nbuffered_per_pe = (unsigned *)malloc(npes * sizeof(unsigned));
    assert(buf->nbuffered_per_pe);
    memset(buf->nbuffered_per_pe, 0x00, npes * sizeof(unsigned));

    buf->active_buffers = (mailbox_header_wrapper_t **)malloc(
            npes * sizeof(buf->active_buffers[0]));
    assert(buf->active_buffers);
    memset(buf->active_buffers, 0x00, npes * sizeof(buf->active_buffers[0]));

    buf->pending_pool_head = NULL;
    buf->pending_pool_tail = NULL;
    buf->free_pool = NULL;

    for (int i = 0; i < buffer_pool_size; i++) {
        shmem_ctx_t ctx;
        mailbox_msg_header_t *msg = mailbox_allocate_msg(
                buffer_size_per_pe * msg_size, &ctx);
        mailbox_header_wrapper_t *wrapper = (mailbox_header_wrapper_t*)malloc(
                sizeof(*wrapper));
        assert(wrapper);

        wrapper->next = buf->free_pool;
        wrapper->prev = NULL;
        wrapper->msg = msg;
        wrapper->ctx = ctx;
        if (buf->free_pool) buf->free_pool->prev = wrapper;
        buf->free_pool = wrapper;
    }
}

static mailbox_header_wrapper_t* pop_pending_buffer(mailbox_buffer_t *buf) {
    mailbox_header_wrapper_t* msg = buf->pending_pool_head;
    assert(msg);
    mailbox_sync(msg->ctx, buf->mbox);

    buf->pending_pool_head = msg->next;
    if (buf->pending_pool_head) {
        buf->pending_pool_head->prev = NULL;
    } else {
        buf->pending_pool_tail = NULL;
    }

    return msg;
}

static mailbox_header_wrapper_t* allocate_buffer(int pe, mailbox_buffer_t *buf) {
    mailbox_header_wrapper_t *msg = NULL;

    if (buf->free_pool) {
        msg = buf->free_pool;
        buf->free_pool = msg->next;
        if (buf->free_pool) {
            buf->free_pool->prev = NULL;
        }
    } else {
        msg = pop_pending_buffer(buf);
    }

    msg->next = msg->prev = NULL;
    msg->pe = pe;
    assert(buf->active_buffers[pe] == NULL);
    buf->active_buffers[pe] = msg;
    return msg;
}

static void add_pending_buffer(mailbox_header_wrapper_t* msg,
        mailbox_buffer_t *buf) {
    assert(buf->active_buffers[msg->pe] == msg);
    buf->active_buffers[msg->pe] = NULL;

    if (buf->pending_pool_head == NULL && buf->pending_pool_tail == NULL) {
        // Empty list
        buf->pending_pool_head = buf->pending_pool_tail = msg;
        msg->next = msg->prev = NULL;
    } else {
        buf->pending_pool_tail->next = msg;
        msg->prev = buf->pending_pool_tail;
        msg->next = NULL;
        buf->pending_pool_tail = msg;
    }
}

int mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, mailbox_buffer_t *buf) {
    assert(msg_len == buf->msg_size);

    unsigned nbuffered = buf->nbuffered_per_pe[target_pe];
    assert(nbuffered <= buf->buffer_size_per_pe);

    if (nbuffered == buf->buffer_size_per_pe) {
        mailbox_header_wrapper_t *pe_buf = buf->active_buffers[target_pe];
        assert(pe_buf);

        // flush
        int success = mailbox_send(pe_buf->msg, pe_buf->ctx,
                nbuffered * buf->msg_size, target_pe, max_tries, buf->mbox);
        if (success) {
            buf->nbuffered_per_pe[target_pe] = 0;
            add_pending_buffer(pe_buf, buf);
        } else {
            return 0;
        }
    }

    if (buf->active_buffers[target_pe] == NULL) {
        allocate_buffer(target_pe, buf);
    }
    mailbox_header_wrapper_t *pe_buf = buf->active_buffers[target_pe];

    nbuffered = buf->nbuffered_per_pe[target_pe];
    char *dst = ((char *)(pe_buf->msg + 1)) + (nbuffered * buf->msg_size);
    memcpy(dst, msg, msg_len);
    buf->nbuffered_per_pe[target_pe] = nbuffered + 1;
    return 1;
}

int mailbox_buffer_flush(mailbox_buffer_t *buf, int max_tries) {
    for (int p = 0; p < buf->npes; p++) {
        const unsigned nbuffered = buf->nbuffered_per_pe[p];
        assert(nbuffered <= buf->buffer_size_per_pe);

        mailbox_header_wrapper_t *pe_buf = buf->active_buffers[p];
        if (nbuffered > 0) {
            assert(pe_buf);
            unsigned count_loops = 0;
            int printed_warning = 0;

            int success = mailbox_send(pe_buf->msg, pe_buf->ctx,
                    nbuffered * buf->msg_size, p, max_tries, buf->mbox);
            if (success) {
                buf->nbuffered_per_pe[p] = 0;
                add_pending_buffer(pe_buf, buf);
            } else {
                return 0;
            }
        }
    }

    while (buf->pending_pool_head) {
        mailbox_header_wrapper_t *iter = pop_pending_buffer(buf);
        iter->prev = NULL;
        iter->next = buf->free_pool;
        buf->free_pool = iter;
    }
    assert(buf->pending_pool_head == NULL && buf->pending_pool_tail == NULL);

    return 1;
}
