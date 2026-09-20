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
#include <time.h>
#include <unistd.h>

#ifndef NODE_SHIFT
#define NODE_SHIFT 12
#endif

#define NVME_CMD_NODEID_WRITE 0x81
#define AGG_SPECIAL_NODEID    0xFFFFFFFFFFFFFFFFULL
#define MAX_NODE_IDS          8

struct ParamEntry {
    uint32_t flag;
    uint32_t hop;
    uint32_t node_num;
    uint64_t node_id[MAX_NODE_IDS];
};

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

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s /dev/ng0n1 <nsid> <node_id0> [node_id1 ... node_id7]\n"
            "Example: %s /dev/ng0n1 1 5 6 7 8\n",
            argv[0], argv[0]);
        return 2;
    }

    const char *dev = argv[1];
    uint32_t nsid = (uint32_t)strtoul(argv[2], NULL, 0);

    uint32_t node_num = (uint32_t)(argc - 3);
    if (node_num == 0 || node_num > MAX_NODE_IDS) {
        fprintf(stderr, "Error: node_num must be 1~%u, got %u\n",
                MAX_NODE_IDS, node_num);
        return 2;
    }

    uint64_t trigger_nodeid = AGG_SPECIAL_NODEID;
    uint32_t nblks = 1;
    uint32_t data_len = 1U << NODE_SHIFT;

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        die("open");
    }

    void *buf = NULL;
    if (posix_memalign(&buf, 1UL << NODE_SHIFT, data_len) != 0) {
        die("posix_memalign");
    }

    memset(buf, 0, data_len);

    struct ParamEntry *param = (struct ParamEntry *)buf;

    param->flag = 1;       /* Batch mode */
    param->hop = 0;        /* unused in batch mode */
    param->node_num = node_num;

    for (uint32_t i = 0; i < node_num; i++) {
        param->node_id[i] = strtoull(argv[3 + i], NULL, 0);
    }

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

    cmd.opcode   = NVME_CMD_NODEID_WRITE;
    cmd.nsid     = nsid;
    cmd.addr     = (uint64_t)(uintptr_t)buf;
    cmd.data_len = data_len;

    cmd.cdw10 = (uint32_t)(trigger_nodeid & 0xffffffffULL);
    cmd.cdw11 = (uint32_t)((trigger_nodeid >> 32) & 0xffffffffULL);
    cmd.cdw12 = nblks & 0xffff;

    memcpy(sqe->cmd, &cmd, sizeof(cmd));
    sqe->user_data = trigger_nodeid;

    uint64_t t_start = now_ns();

    ret = io_uring_submit(&ring);
    if (ret < 0) {
        io_uring_queue_exit(&ring);
        free(buf);
        close(fd);
        die_uring("io_uring_submit", ret);
    }

    struct io_uring_cqe *cqe;
    ret = io_uring_wait_cqe(&ring, &cqe);

    uint64_t t_end = now_ns();
    uint64_t latency_ns = t_end - t_start;

    if (ret < 0) {
        io_uring_queue_exit(&ring);
        free(buf);
        close(fd);
        die_uring("io_uring_wait_cqe", ret);
    }

    if (cqe->res < 0) {
        errno = -cqe->res;
        fprintf(stderr,
                "Batch aggregation cmd failed: res=%d errno=%d (%s)\n",
                cqe->res, errno, strerror(errno));
    } else {
        printf("SUCCESS: submitted BATCH aggregation command\n");
        printf("  flag     = %u\n", param->flag);
        printf("  hop      = %u\n", param->hop);
        printf("  node_num = %u\n", param->node_num);

        for (uint32_t i = 0; i < param->node_num; i++) {
            printf("  node_id[%u] = %" PRIu64 "\n", i, param->node_id[i]);
        }

        printf("  trigger_nodeid = 0x%016" PRIx64 "\n", trigger_nodeid);
    }

    printf("\n=== Host-side latency ===\n");
    printf("host_latency_ns = %" PRIu64 "\n", latency_ns);
    printf("host_latency_us = %.3f us\n", latency_ns / 1000.0);
    printf("host_latency_ms = %.3f ms\n", latency_ns / 1000000.0);

    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);

    free(buf);
    close(fd);

    return 0;
}