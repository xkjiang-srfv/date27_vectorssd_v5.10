 


#ifndef RAW_REGION_H
#define RAW_REGION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../nvme.h"


#define MAX_NODE_IDS 16
#define MAX_DISKANN_NODES ((uint64_t)0x8000000ULL)  // 134,217,728


#define AGG_CMD_MAGIC           0x41474743
#define AGG_RES_MAGIC           0x41474752

struct ParamEntry {
    uint32_t flag;
    uint32_t hop;
    uint32_t node_num;
    uint64_t node_id[MAX_NODE_IDS];
};


typedef enum {
    RAW_QUERY_DFS = 0,
    RAW_QUERY_BATCH = 1,
} raw_query_type_t;


typedef struct {
    uint32_t magic;
    uint32_t status;
    uint64_t result_size;
    uint64_t total_latency;
    uint64_t read_latency;
    uint64_t write_latency;
    uint32_t read_cnt;
    uint32_t write_cnt;
} agg_result_meta_t;

#endif