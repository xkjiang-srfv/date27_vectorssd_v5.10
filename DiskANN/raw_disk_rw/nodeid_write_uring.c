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

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s /dev/ng0n1 <nsid> <nodeid> <nblks>\n"
            "Example: %s /dev/ng0n1 1 1234 2\n",
            argv[0], argv[0]);
        return 2;
    }

    const char *dev = argv[1];
    uint32_t nsid   = (uint32_t)strtoul(argv[2], NULL, 0);
    uint64_t nodeid = (uint64_t)strtoull(argv[3], NULL, 0);
    uint32_t nblks  = (uint32_t)strtoul(argv[4], NULL, 0);

    if (nblks == 0) {
        fprintf(stderr, "nblks must be > 0\n");
        return 2;
    }

    if (nblks > 65535) {
        fprintf(stderr, "nblks too large for FEMU field, cdw12 low16 nlb: %u\n",
                nblks);
        return 2;
    }

    size_t data_len_sz = ((size_t)nblks) << NODE_SHIFT;

    if (data_len_sz > UINT32_MAX) {
        fprintf(stderr, "data_len too large: %zu\n", data_len_sz);
        return 2;
    }

    uint32_t data_len = (uint32_t)data_len_sz;

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        die("open");
    }

    void *buf = NULL;
    if (posix_memalign(&buf, 1UL << NODE_SHIFT, data_len) != 0) {
        die("posix_memalign");
    }

    memset(buf, 0xAB, data_len);

    struct io_uring ring;
    struct io_uring_params p;

    memset(&p, 0, sizeof(p));

     



    p.flags |= IORING_SETUP_SQE128;
    p.flags |= IORING_SETUP_CQE32;

    int ret = io_uring_queue_init_params(8, &ring, &p);
    if (ret < 0) {
        free(buf);
        close(fd);
        die_uring("io_uring_queue_init_params", ret);
    }

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
        io_uring_queue_exit(&ring);
        free(buf);
        close(fd);
        die("io_uring_get_sqe");
    }

     



    io_uring_prep_uring_cmd(sqe, NVME_URING_CMD_IO, fd);

    struct nvme_uring_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.opcode   = 0x81;   // NVME_CMD_NODEID_WRITE
    cmd.nsid     = nsid;
    cmd.addr     = (uint64_t)(uintptr_t)buf;
    cmd.data_len = data_len;

     








    cmd.cdw10 = (uint32_t)(nodeid & 0xffffffffULL);
    cmd.cdw11 = (uint32_t)((nodeid >> 32) & 0xffffffffULL);
    cmd.cdw12 = nblks & 0xffff;

     


    memcpy(sqe->cmd, &cmd, sizeof(cmd));

    sqe->user_data = nodeid;

    ret = io_uring_submit(&ring);
    if (ret < 0) {
        io_uring_queue_exit(&ring);
        free(buf);
        close(fd);
        die_uring("io_uring_submit", ret);
    }

    struct io_uring_cqe *cqe;

    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
        io_uring_queue_exit(&ring);
        free(buf);
        close(fd);
        die_uring("io_uring_wait_cqe", ret);
    }

    if (cqe->res < 0) {
        errno = -cqe->res;
        fprintf(stderr,
                "NVMe uring cmd failed: res=%d errno=%d (%s)\n",
                cqe->res, errno, strerror(errno));
    } else {
        printf("SUCCESS: wrote %u * 4KB to nodeid=%" PRIu64 " "
               "(nsid=%u, cdw10=0x%08x, cdw11=0x%08x)\n",
               nblks,
               nodeid,
               nsid,
               cmd.cdw10,
               cmd.cdw11);
    }

    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);

    free(buf);
    close(fd);

    return 0;
}