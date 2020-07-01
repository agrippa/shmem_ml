#ifndef _MAILBOX_H
#define _MAILBOX_H

#include <stdint.h>
#include <stdlib.h>

typedef struct _mailbox_t {
    uint64_t *indices;
    uint64_t indices_curr_val;
    uint32_t capacity_in_bytes;
    char *buf;
    int pe;
} mailbox_t;

/*
 * A symmetric call that allocates a remotely accessible mailbox data structure
 * across all PEs.
 */
void mailbox_init(mailbox_t *mailbox, size_t capacity_in_bytes);

/*
 * Place msg with length msg_len in bytes into the designated mailbox on the
 * designated PE. Will retry max_tries time if the mailbox does not have enough
 * space, or infinitely if max_tries is set to -1. Returns 1 if the send
 * succeeds, 0 otherwise.
 */
int mailbox_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, mailbox_t *mailbox);

/*
 * Check my local mailbox for a new message. If one is found, a pointer to it is
 * stored in msg, msg_len is updated to reflect the length of the message, and 1
 * is returned. Otherwise, 0 is returned to indicate no message was found.
 */
int mailbox_recv(void *msg, size_t msg_capacity, size_t *msg_len,
        mailbox_t *mailbox);

void mailbox_destroy(mailbox_t *mailbox);

#endif // _HVR_MAILBOX
