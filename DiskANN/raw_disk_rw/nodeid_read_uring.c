#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <liburing.h>
#include <linux/nvme_ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef NODE_SHIFT
#define NODE_SHIFT 12  // 4KB
#endif

static void die(const char *msg)
{
    perror(msg);
    exit(1);
}

static void die_uring(const char *msg, int ret)
{
    errno = -ret;
    perror(msg);
    exit(1);
}

static void hexdump_prefix(const void *buf, size_t len, size_t max_bytes)
{
    const unsigned char *p = (const unsigned char *)buf;
    size_t n = len < max_bytes ? len : max_bytes;

    printf("First %zu bytes:\n", n);
    for (size_t i = 0; i < n; i++) {
        printf("%02X ", p[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    if (n % 16 != 0)
        printf("\n");
}

static int verify_fill(const void *buf, size_t len, unsigned char expected)
{
    const unsigned char *p = (const unsigned char *)buf;
    for (size_t i = 0; i < len; i++) {
        if (p[i] != expected)
            return (int)i;  // mismatch index
    }
    return -1; // ok
}

int main(int argc, char **argv)
{
    int verify_ab = 0;

    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s /dev/ng0n1 <nsid> <nodeid> <nblks> [--verify-ab]\n"
            "Example: %s /dev/ng0n1 1 1234 2 --verify-ab\n",
            argv[0], argv[0]);
        return 2;
    }

    /* optional flag */
    if (argc >= 6 && strcmp(argv[5], "--verify-ab") == 0)
        verify_ab = 1;

    const char *dev = argv[1];
    uint32_t nsid   = (uint32_t)strtoul(argv[2], NULL, 0);
    uint32_t nodeid = (uint32_t)strtoul(argv[3], NULL, 0);
    uint32_t nblks  = (uint32_t)strtoul(argv[4], NULL, 0);

    if (nblks == 0) {
        fprintf(stderr, "nblks must be > 0\n");
        return 2;
    }
    if (nblks > 65535) {
        fprintf(stderr, "nblks too large for FEMU field (u16 nlb): %u\n", nblks);
        return 2;
    }

    size_t data_len_sz = ((size_t)nblks) << NODE_SHIFT;
    if (data_len_sz > UINT32_MAX) {
        fprintf(stderr, "data_len too large: %zu\n", data_len_sz);
        return 2;
    }
    uint32_t data_len = (uint32_t)data_len_sz;

    int fd = open(dev, O_RDWR);
    if (fd < 0)
        die("open");

    void *buf = NULL;
    if (posix_memalign(&buf, 1UL << NODE_SHIFT, data_len) != 0)
        die("posix_memalign");

     
    memset(buf, 0, data_len);

    struct io_uring ring;
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));

     
    p.flags |= IORING_SETUP_SQE128;
    p.flags |= IORING_SETUP_CQE32;

    int ret = io_uring_queue_init_params(8, &ring, &p);
    if (ret < 0)
        die_uring("io_uring_queue_init_params", ret);

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    if (!sqe)
        die("io_uring_get_sqe");

     
    io_uring_prep_uring_cmd(sqe, NVME_URING_CMD_IO, fd);

     
    struct nvme_uring_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.opcode   = 0x82;   
    cmd.nsid     = nsid;
    cmd.addr     = (uint64_t)(uintptr_t)buf;
    cmd.data_len = data_len;

    /* FEMU NvmeNodeCmd: nodeid = cdw10+cdw11, nlb = low16(cdw12) */
    cmd.cdw10 = nodeid;
    cmd.cdw11 = 0;
    cmd.cdw12 = nblks;

    memcpy(sqe->cmd, &cmd, sizeof(cmd));
    sqe->user_data = (uint64_t)nodeid;

    ret = io_uring_submit(&ring);
    if (ret < 0)
        die_uring("io_uring_submit", ret);

    struct io_uring_cqe *cqe;
    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0)
        die_uring("io_uring_wait_cqe", ret);

    if (cqe->res < 0) {
        errno = -cqe->res;
        fprintf(stderr, "NVMe uring READ failed: res=%d errno=%d (%s)\n",
                cqe->res, errno, strerror(errno));
        io_uring_cqe_seen(&ring, cqe);
        io_uring_queue_exit(&ring);
        free(buf);
        close(fd);
        return 1;
    }

    printf("READ SUCCESS: read %u * 4KB from nodeid=%u (nsid=%u)\n",
           nblks, nodeid, nsid);

    hexdump_prefix(buf, data_len, 64);

    if (verify_ab) {
        int bad = verify_fill(buf, data_len, 0xAB);
        if (bad < 0) {
            printf("VERIFY PASS: all bytes are 0xAB\n");
        } else {
            printf("VERIFY FAIL: mismatch at offset %d: got 0x%02X expected 0xAB\n",
                   bad, ((unsigned char*)buf)[bad]);
        }
    }

    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);

    free(buf);
    close(fd);
    return 0;
}