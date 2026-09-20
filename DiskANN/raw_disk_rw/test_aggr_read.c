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

#define NVME_CMD_AGGRE_READ 0x83
#define PAGE_SIZE 4096
#define MAX_NODE_IDS 8
#define AGG_RES_MAGIC 0x41474752

#define DISKANN_DIM 1024
#define DISKANN_MAX_DEGREE 128

struct ParamEntry {
    uint32_t flag;
    uint32_t hop;
    uint32_t node_num;
    uint64_t node_id[MAX_NODE_IDS];
};

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
    uint64_t node_id;
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
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s /dev/ng0n1 <nsid> <result_nblks> <nodeid0> [nodeid1 ... nodeid7]\n",
            argv[0]);
        return 1;
    }

    const char *dev = argv[1];
    uint32_t nsid = (uint32_t)strtoul(argv[2], NULL, 0);
    uint32_t result_nblks = (uint32_t)strtoul(argv[3], NULL, 0);

    uint32_t node_num = argc - 4;
    if (node_num == 0 || node_num > MAX_NODE_IDS) {
        fprintf(stderr, "node_num must be 1..8\n");
        return 1;
    }

    uint64_t result_size = (uint64_t)result_nblks * PAGE_SIZE;

    struct ParamEntry *param = NULL;
    void *result_buf = NULL;

    if (posix_memalign((void **)&param, PAGE_SIZE, PAGE_SIZE) != 0) {
        die("posix_memalign param");
    }

    if (posix_memalign(&result_buf, PAGE_SIZE, result_size) != 0) {
        die("posix_memalign result");
    }

    memset(param, 0, PAGE_SIZE);
    memset(result_buf, 0, result_size);

    param->flag = 1;
    param->hop = 0;
    param->node_num = node_num;

    for (uint32_t i = 0; i < node_num; i++) {
        param->node_id[i] = strtoull(argv[4 + i], NULL, 0);
        printf("param node[%u] = %" PRIu64 "\n", i, param->node_id[i]);
    }

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        die("open");
    }

    struct io_uring ring;
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));

    p.flags |= IORING_SETUP_SQE128;
    p.flags |= IORING_SETUP_CQE32;

    int ret = io_uring_queue_init_params(8, &ring, &p);
    if (ret < 0) {
        die_uring("io_uring_queue_init_params", ret);
    }

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
        die("io_uring_get_sqe");
    }

    io_uring_prep_uring_cmd(sqe, NVME_URING_CMD_IO, fd);

    struct nvme_uring_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.opcode = NVME_CMD_AGGRE_READ;
    cmd.nsid = nsid;

     



    cmd.addr = (uint64_t)(uintptr_t)result_buf;
    cmd.data_len = result_size;

     



    cmd.metadata = (uint64_t)(uintptr_t)param;
    cmd.metadata_len = sizeof(struct ParamEntry);

     




    cmd.cdw10 = result_nblks;
    cmd.cdw11 = 0;

    memcpy(sqe->cmd, &cmd, sizeof(cmd));
    sqe->user_data = 0x83;

    ret = io_uring_submit(&ring);
    if (ret < 0) {
        die_uring("io_uring_submit", ret);
    }

    struct io_uring_cqe *cqe = NULL;
    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
        die_uring("io_uring_wait_cqe", ret);
    }

    if (cqe->res < 0) {
        errno = -cqe->res;
        fprintf(stderr, "AGGRE_READ failed: res=%d errno=%d %s\n",
                cqe->res, errno, strerror(errno));
        goto out;
    }

    struct agg_result_meta *meta = (struct agg_result_meta *)result_buf;

    printf("\n=== Aggregation Result Meta ===\n");
    printf("magic         = 0x%08x\n", meta->magic);
    printf("status        = %u\n", meta->status);
    printf("result_size   = %" PRIu64 "\n", meta->result_size);
    printf("total_latency = %" PRIu64 "\n", meta->total_latency);
    printf("read_latency  = %" PRIu64 "\n", meta->read_latency);
    printf("read_cnt      = %u\n", meta->read_cnt);

    if (meta->magic != AGG_RES_MAGIC) {
        fprintf(stderr, "invalid magic\n");
        goto out;
    }

    if (meta->status != 0) {
        fprintf(stderr, "aggregation failed in FEMU\n");
        goto out;
    }

    if (sizeof(*meta) + meta->result_size > result_size) {
        fprintf(stderr, "result buffer too small\n");
        goto out;
    }

    uint64_t node_cnt = meta->result_size / sizeof(struct diskann_prefetch_node);

    printf("\nnode_cnt = %" PRIu64 "\n", node_cnt);

    struct diskann_prefetch_node *nodes =
        (struct diskann_prefetch_node *)((uint8_t *)result_buf + sizeof(*meta));

    for (uint64_t i = 0; i < node_cnt; i++) {
        printf("\nnode[%" PRIu64 "] id=%lu degree=%u\n",
               i, nodes[i].node_id, nodes[i].degree);

        printf("vector[0..3] = %.4f %.4f %.4f %.4f\n",
               nodes[i].vector[0],
               nodes[i].vector[1],
               nodes[i].vector[2],
               nodes[i].vector[3]);

        printf("neighbors[0..7] = ");
        for (uint32_t j = 0; j < nodes[i].degree && j < 8; j++) {
            printf("%u ", nodes[i].neighbors[j]);
        }
        printf("\n");
    }

out:
    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);
    close(fd);
    free(param);
    free(result_buf);
    return 0;
}