#ifndef DISKANN_FORMAT_H
#define DISKANN_FORMAT_H

#include <stdint.h>

#define DISKANN_DIM 100
#define DISKANN_MAX_DEGREE 64

typedef int8_t diskann_vec_t;

#pragma pack(push, 1)
typedef struct {
    diskann_vec_t vector[DISKANN_DIM];
    uint32_t neighbor_count;
    uint32_t neighbors[DISKANN_MAX_DEGREE];
} diskann_node_t;
#pragma pack(pop)

#define DISKANN_NODE_VEC_BYTES   (DISKANN_DIM * sizeof(diskann_vec_t))
#define DISKANN_NODE_NBRS_BYTES  (DISKANN_MAX_DEGREE * sizeof(uint32_t))
#define DISKANN_NODE_BYTES       ((uint64_t)sizeof(diskann_node_t))

#endif