#include <shmem.h>
#include <shmemx.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "mailbox.hpp"

#define CRC32
#include "crc.c"

void mailbox_init(mailbox_t *mailbox, size_t capacity_in_bytes) {
    memset(mailbox, 0x00, sizeof(*mailbox));
    mailbox->indices = (uint64_t *)shmem_malloc(
            sizeof(*(mailbox->indices)));
    assert(mailbox->indices);
    shmem_uint64_p(mailbox->indices, 0, shmem_my_pe());
    mailbox->indices_curr_val = 0;

    mailbox->capacity_in_bytes = capacity_in_bytes;

    mailbox->buf = (char *)shmem_malloc(capacity_in_bytes);
    assert(mailbox->buf);
    memset(mailbox->buf, 0x00, capacity_in_bytes);

    mailbox->pe = shmem_my_pe();

    crcInit();

    shmem_barrier_all();
}

static uint64_t pack_indices(uint32_t read_index, uint32_t write_index) {
    uint64_t packed = read_index;
    packed = (packed << 32);
    packed = (packed & 0xffffffff00000000);
    packed = (packed | (uint64_t)write_index);
    return packed;
}

static void unpack_indices(uint64_t index, uint32_t *read_index,
        uint32_t *write_index) {
    *write_index = (index & 0x00000000ffffffff);
    uint64_t tmp = (index >> 32);
    *read_index = tmp;
}

static uint32_t used_bytes(uint32_t read_index, uint32_t write_index,
        mailbox_t *mailbox) {
    if (write_index >= read_index) {
        return write_index - read_index;
    } else {
        return write_index + (mailbox->capacity_in_bytes - read_index);
    }
}

static void clear_mailbox_with_rotation(size_t data_len,
        uint64_t starting_offset, mailbox_t *mailbox) {
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        memset(mailbox->buf + starting_offset, 0x00, data_len);
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        memset(mailbox->buf + starting_offset, 0x00, rotate_index);
        memset(mailbox->buf, 0x00, data_len - rotate_index);
    }
}

static void put_in_mailbox_with_rotation(const void *data, size_t data_len,
        uint64_t starting_offset, mailbox_t *mailbox, int target_pe,
        shmem_ctx_t ctx) {
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        shmem_ctx_putmem_nbi(ctx, mailbox->buf + starting_offset, data, data_len, target_pe);
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        shmem_ctx_putmem_nbi(ctx, mailbox->buf + starting_offset, data, rotate_index,
                target_pe);
        shmem_ctx_putmem_nbi(ctx, mailbox->buf, (char *)data + rotate_index,
                data_len - rotate_index, target_pe);
    }
}

static void get_from_mailbox_with_rotation(uint64_t starting_offset, void *data,
        uint64_t data_len, mailbox_t* mailbox) {
    const int target_pe = mailbox->pe;
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        shmem_getmem(data, mailbox->buf + starting_offset, data_len, target_pe);
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        shmem_getmem(data, mailbox->buf + starting_offset, rotate_index,
                target_pe);
        shmem_getmem((char *)data + rotate_index, mailbox->buf,
                data_len - rotate_index, target_pe);
    }
}

mailbox_msg_header_t* mailbox_allocate_msg(size_t max_msg_len,
        shmem_ctx_t *out_ctx) {
    mailbox_msg_header_t* msg = (mailbox_msg_header_t*)malloc(
            sizeof(*msg) + max_msg_len);
    assert(msg);

    int err = shmem_ctx_create(0, out_ctx);
    assert(err == 0);

    return msg;
}

void mailbox_deallocate_msg(mailbox_msg_header_t* msg, shmem_ctx_t ctx) {
    free(msg);
    shmem_ctx_destroy(ctx);
}

void mailbox_sync(shmem_ctx_t ctx, mailbox_t *mailbox) {
    shmem_ctx_quiet(ctx);
}

int mailbox_send(mailbox_msg_header_t *msg, shmem_ctx_t ctx, size_t msg_len,
        int target_pe, int max_tries, mailbox_t *mailbox) {
    // So that sentinel values are always cohesive

    uint64_t full_msg_len = sizeof(*msg) + msg_len;
    // Hash the length of the message
    msg->msg_len_crc = crcFast((const unsigned char *)&msg_len,
            sizeof(msg_len));
    // Hash the message itself
    msg->msg_crc = crcFast((const unsigned char *)(msg + 1), msg_len);
    msg->msg_len = msg_len;

    assert(full_msg_len < mailbox->capacity_in_bytes);

    uint64_t indices = shmem_ctx_uint64_atomic_fetch(ctx, mailbox->indices, target_pe);
    uint32_t start_send_index = 0;

    unsigned tries = 0;
    while (max_tries < 0 || tries < max_tries) {
        if (tries > 1000000) {
            fprintf(stderr, "WARNING PE %d hitting many failed tries sending "
                    "to %d\n", shmem_my_pe(), target_pe);
            abort();
        }
        uint32_t read_index, write_index;
        unpack_indices(indices, &read_index, &write_index);

        uint32_t consumed = used_bytes(read_index, write_index, mailbox);
        uint32_t free_bytes = mailbox->capacity_in_bytes - consumed;
        if (free_bytes > full_msg_len) {
            // Enough room to try
            uint32_t new_write_index = (write_index + full_msg_len) %
                mailbox->capacity_in_bytes;
            uint64_t new_val = pack_indices(read_index, new_write_index);
            uint64_t old = shmem_ctx_uint64_atomic_compare_swap(ctx, mailbox->indices,
                    indices, new_val, target_pe);
            if (old == indices) {
                // Successful
                start_send_index = write_index;
                break;
            } else {
                indices = old;
            }
        } else {
            indices = shmem_ctx_uint64_atomic_fetch(ctx, mailbox->indices, target_pe);
        }
        tries++;
    }

    if (tries == max_tries) {
        // Failed
        return 0;
    }

    /*
     * Send the actual message, accounting for if the space allocated goes
     * around the circular buffer.
     */
    put_in_mailbox_with_rotation(msg, full_msg_len,
            start_send_index, mailbox, target_pe, ctx);
    return 1;
}

int mailbox_recv(void *msg, size_t msg_capacity, size_t *msg_len,
        mailbox_t *mailbox) {
    uint32_t read_index, write_index;
    uint64_t curr_indices;

    unpack_indices(mailbox->indices_curr_val, &read_index, &write_index);
    if (used_bytes(read_index, write_index, mailbox) > 0) {
        /*
         * If the previously saved current value of indices indicates there are
         * pending messages, we can assume that is still the case without
         * actually having to check the mailbox.
         */
        curr_indices = mailbox->indices_curr_val;
    } else {
        /*
         * Otherwise, the last time we checked the mailbox it was empty. We have
         * to check if that's still the case.
         */
        uint64_t new_indices = shmem_uint64_atomic_fetch(mailbox->indices,
                    mailbox->pe);
        if (new_indices != mailbox->indices_curr_val) {
            curr_indices = new_indices;
        } else {
            return 0;
        }
    }

    unpack_indices(curr_indices, &read_index, &write_index);
    uint32_t used = used_bytes(read_index, write_index, mailbox);
    assert(used > 0);

    // Wait for the sentinel value to appear
    mailbox_msg_header_t header;
    uint64_t header_offset = read_index % mailbox->capacity_in_bytes;
    uint64_t msg_offset = ((read_index + sizeof(header)) %
            mailbox->capacity_in_bytes);
    get_from_mailbox_with_rotation(header_offset, &header, sizeof(header),
            mailbox);

    crc calc_msg_len_crc = crcFast((const unsigned char *)&header.msg_len,
            sizeof(header.msg_len));
    if (calc_msg_len_crc != header.msg_len_crc) {
        // Checksums for message length don't match
        return 0;
    }

    *msg_len = header.msg_len;
    assert(msg_capacity >= header.msg_len);
    get_from_mailbox_with_rotation(msg_offset, msg, header.msg_len, mailbox);

    crc calc_msg_crc = crcFast((const unsigned char *)msg, header.msg_len);
    if (calc_msg_crc != header.msg_crc) {
        // Checksums for message don't match
        return 0;
    }

    /*
     * Once we've finished extracting the message, clear the checksums to make
     * sure we don't accidentally detect this message again.
     */
    clear_mailbox_with_rotation(sizeof(header), header_offset, mailbox);
    shmem_quiet();

    uint32_t new_read_index = (msg_offset + header.msg_len) %
        mailbox->capacity_in_bytes;
    uint64_t new_indices = pack_indices(new_read_index, write_index);
    while (1) {
        uint64_t old = shmem_uint64_atomic_compare_swap(mailbox->indices,
                curr_indices, new_indices, mailbox->pe);
        if (old == curr_indices) break;

        uint32_t this_read_index, this_write_index;
        unpack_indices(old, &this_read_index, &this_write_index);
        assert(read_index == this_read_index);

        curr_indices = old;
        new_indices = pack_indices(new_read_index, this_write_index);
    }
    mailbox->indices_curr_val = new_indices;

    return 1;
}
