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
#define NODE_SHIFT 12
#endif

#define NVME_CMD_NODEID_READ 0x82
#define AGG_SPECIAL_NODEID   0xFFFFFFFFFFFFFFFFULL
#define AGG_RES_MAGIC        0x41474752

#define DISKANN_DIM          1024
#define DISKANN_MAX_DEGREE   128

struct agg_result_meta {
    uint32_t magic;
    uint32_t status;
    uint64_t result_size;
    uint64_t total_latency;
    uint64_t read_latency;
    uint64_t write_latency;
    uint32_t read_cnt;
    uint32_t write_cnt;
};

struct diskann_prefetch_node {
    uint32_t node_id;
    uint32_t degree;
    float vector[DISKANN_DIM];
    uint32_t neighbors[DISKANN_MAX_DEGREE];
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

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s /dev/ng0n1 <nsid>\n"
                "Example: %s /dev/ng0n1 1\n",
                argv[0], argv[0]);
        return 2;
    }

    const char *dev = argv[1];
    uint32_t nsid = (uint32_t)strtoul(argv[2], NULL, 0);

    uint64_t nodeid = AGG_SPECIAL_NODEID;

     









    uint32_t nblks = 16;
    uint32_t data_len = nblks << NODE_SHIFT;

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        die("open");
    }

    void *buf = NULL;
    if (posix_memalign(&buf, 1UL << NODE_SHIFT, data_len) != 0) {
        close(fd);
        die("posix_memalign");
    }

    memset(buf, 0, data_len);

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

    cmd.opcode   = NVME_CMD_NODEID_READ;
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
                "Batch aggregation read failed: res=%d errno=%d (%s)\n",
                cqe->res, errno, strerror(errno));
        goto out;
    }

    struct agg_result_meta *meta = (struct agg_result_meta *)buf;

    printf("=== Aggregation Result Meta ===\n");
    printf("magic         = 0x%08x\n", meta->magic);
    printf("status        = %u\n", meta->status);
    printf("result_size   = %" PRIu64 "\n", meta->result_size);
    printf("total_latency = %" PRIu64 " ns\n", meta->total_latency);
    printf("read_latency  = %" PRIu64 " ns\n", meta->read_latency);
    printf("write_latency = %" PRIu64 " ns\n", meta->write_latency);
    printf("read_cnt      = %u\n", meta->read_cnt);
    printf("write_cnt     = %u\n", meta->write_cnt);

    if (meta->magic != AGG_RES_MAGIC) {
        fprintf(stderr, "Invalid result magic\n");
        goto out;
    }

    if (meta->status != 0) {
        fprintf(stderr, "Aggregation failed in FEMU\n");
        goto out;
    }

    if (meta->result_size == 0) {
        fprintf(stderr, "Empty aggregation result\n");
        goto out;
    }

    if (sizeof(struct agg_result_meta) + meta->result_size > data_len) {
        fprintf(stderr,
                "Result buffer too small: need=%" PRIu64 ", have=%u\n",
                sizeof(struct agg_result_meta) + meta->result_size,
                data_len);
        goto out;
    }

    if (meta->result_size % sizeof(struct diskann_prefetch_node) != 0) {
        fprintf(stderr,
                "Invalid result_size: %" PRIu64 ", entry_size=%zu\n",
                meta->result_size,
                sizeof(struct diskann_prefetch_node));
        goto out;
    }

    uint64_t entry_cnt =
        meta->result_size / sizeof(struct diskann_prefetch_node);

    printf("\n=== Prefetch Nodes ===\n");
    printf("entry_cnt = %" PRIu64 "\n", entry_cnt);
    printf("entry_size = %zu bytes\n", sizeof(struct diskann_prefetch_node));

    struct diskann_prefetch_node *nodes =
        (struct diskann_prefetch_node *)((uint8_t *)buf + sizeof(struct agg_result_meta));

    for (uint64_t i = 0; i < entry_cnt; i++) {
        printf("\nEntry[%" PRIu64 "] node_id=%u degree=%u\n",
               i, nodes[i].node_id, nodes[i].degree);

        printf("vector[0..7]: ");
        for (uint32_t j = 0; j < 8 && j < DISKANN_DIM; j++) {
            printf("%.4f ", nodes[i].vector[j]);
        }
        printf("\n");

        uint32_t deg = nodes[i].degree;
        if (deg > DISKANN_MAX_DEGREE) {
            deg = DISKANN_MAX_DEGREE;
        }

        printf("neighbors: ");
        for (uint32_t j = 0; j < deg; j++) {
            printf("%u ", nodes[i].neighbors[j]);
        }
        printf("\n");
    }

out:
    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);

    free(buf);
    close(fd);

    return 0;
}