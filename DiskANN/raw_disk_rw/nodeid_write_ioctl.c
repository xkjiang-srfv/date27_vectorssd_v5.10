#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/nvme_ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#ifndef NODE_SHIFT
#define NODE_SHIFT 12
#endif

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s /dev/nvme0 <nsid> <nodeid> <nblks>\n"
            "Example: %s /dev/nvme0 1 1234 2\n",
            argv[0], argv[0]);
        return 2;
    }

    const char *dev = argv[1];
    uint32_t nsid   = (uint32_t)strtoul(argv[2], NULL, 0);
    uint32_t nodeid = (uint32_t)strtoul(argv[3], NULL, 0);
    uint32_t nblks  = (uint32_t)strtoul(argv[4], NULL, 0);

    if (nblks == 0 || nblks > 65535) {
        fprintf(stderr, "nblks must be in [1, 65535]\n");
        return 2;
    }

    size_t data_len_sz = ((size_t)nblks) << NODE_SHIFT;
    if (data_len_sz > UINT32_MAX) {
        fprintf(stderr, "data_len too large: %zu\n", data_len_sz);
        return 2;
    }
    uint32_t data_len = (uint32_t)data_len_sz;

    int fd = open(dev, O_RDWR);
    if (fd < 0) die("open");

    void *buf = NULL;
    if (posix_memalign(&buf, 1UL << NODE_SHIFT, data_len) != 0)
        die("posix_memalign");

    memset(buf, 0xAB, data_len);

    struct nvme_passthru_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.opcode = 0x81;            
    cmd.nsid   = nsid;
    cmd.addr   = (uint64_t)(uintptr_t)buf;
    cmd.data_len = data_len;

    cmd.cdw10  = nodeid;          
    cmd.cdw11  = 0;
    cmd.cdw12  = (uint16_t)nblks;  

     
    if (ioctl(fd, NVME_IOCTL_IO_CMD, &cmd) < 0) {
        fprintf(stderr, "ioctl NVME_IOCTL_IO_CMD failed: errno=%d (%s)\n",
                errno, strerror(errno));
        close(fd);
        free(buf);
        return 1;
    }

    printf("SUCCESS(ioctl): wrote %u * 4KB to nodeid=%u (nsid=%u)\n",
           nblks, nodeid, nsid);

    close(fd);
    free(buf);
    return 0;
}