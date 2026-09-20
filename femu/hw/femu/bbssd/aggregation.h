#ifndef __AGGREGATION_H
#define __AGGREGATION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "../nvme.h"
#include "raw_region.h"
#include "diskann_format.h"
#include "lftl.h"

#define MAX_QUERY_READS 16384
#define DISKANN_DATA_NODEID_BASE 1ULL

#ifndef DIV_ROUND_UP

#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))

#endif


typedef struct {
    uint64_t node_id;
    uint32_t degree;
    uint32_t neighbors[DISKANN_MAX_DEGREE];
} diskann_neighbor_list_t;

typedef struct {
    uint64_t node_id;
    uint32_t degree;
    diskann_vec_t vector[DISKANN_DIM];
    uint32_t neighbors[DISKANN_MAX_DEGREE];
} diskann_prefetch_node_t;


typedef struct {
    uint8_t *bits;
} visited_set_t;


typedef enum {
    QUERY_IO_READ,
    QUERY_IO_WRITE,
} query_io_type_t;

typedef struct {
    query_io_type_t type;
    uint64_t node_id_or_lba;
    uint32_t nlb;
    uint64_t issue_ts;
    uint64_t latency;
} query_io_t;

 
typedef struct {
    uint64_t query_id;               
    uint32_t num_hops;               
    visited_set_t visited;           
    uint64_t query_start_ts;         
    query_io_t reads[MAX_QUERY_READS];   
    uint32_t read_cnt;                   
} femu_query_ctx_t;

typedef struct {
    uint64_t total_read_lat;
    uint64_t total_write_lat;
    uint64_t total_lat;
    uint32_t read_cnt;
    uint32_t write_cnt;

    uint8_t *result_buf;
    uint64_t result_size;
    uint64_t result_capacity;
} femu_query_result_t;



bool femu_execute_query_dfs(
    FemuCtrl *n,
    uint32_t max_hop,
    uint64_t root_node_id,
    femu_query_result_t *result
);

bool femu_execute_query_batch(
    FemuCtrl *n,
    uint64_t query_start_ts,
    uint64_t *node_ids,
    uint32_t node_num,
    femu_query_result_t *result
);

bool femu_expand_and_collect_neighbors(
    FemuCtrl *n,
    femu_query_ctx_t *qctx,
    uint64_t root_node_id,
    diskann_neighbor_list_t *out,
    uint32_t *out_cnt
);

bool femu_read_diskann_node(
    FemuCtrl *n,
    femu_query_ctx_t *qctx,
    uint64_t node_id,
    diskann_node_t **node_out
);

void femu_dump_diskann_node(
    FemuCtrl *n,
    uint64_t node_id
);

bool femu_execute_aggregation_param(
    FemuCtrl *n,
    struct ParamEntry *cmd,
    void *result_buf,
    uint64_t result_buf_size,
    uint64_t *lat_out);

uint16_t nvme_write_result_prp(
    FemuCtrl *n,
    NvmeRequest *req,
    uint64_t prp1,
    uint64_t prp2,
    void *buf,
    uint64_t len);

#endif