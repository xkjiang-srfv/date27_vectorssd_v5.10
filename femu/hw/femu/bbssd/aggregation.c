#include "aggregation.h"
#include "raw_region.h"
#include "diskann_format.h"
#include "lftl.h"
#include "../nvme.h"

#include <stdlib.h>
#include <string.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef DIV_ROUND_UP
#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))
#endif


static inline uint64_t logical_to_ssd_nodeid(uint64_t logical_id)
{
    return logical_id + DISKANN_DATA_NODEID_BASE;
}

static inline uint32_t ssd_to_logical_nodeid(uint64_t ssd_nodeid)
{
    return (uint32_t)(ssd_nodeid - DISKANN_DATA_NODEID_BASE);
}


static inline void visited_init(visited_set_t *v)
{
    v->bits = calloc(DIV_ROUND_UP(MAX_DISKANN_NODES, 8), 1);
}

static bool result_append_lists(femu_query_result_t *result,diskann_neighbor_list_t *lists,uint32_t list_cnt)
{
    uint64_t payload_limit;
    uint64_t remain;
    uint64_t max_entries;
    uint64_t append_size;
    uint64_t new_size;
    uint8_t *new_buf;

    femu_log("[AGG][APPEND] enter result=%p lists=%p list_cnt=%u\n",result,lists,list_cnt);
    if (!result || !lists || list_cnt == 0) {
        femu_log("[AGG][APPEND] skip invalid input\n");
        return true;
    }

    payload_limit = result->result_capacity;

    femu_log("[AGG][APPEND] payload_limit=%lu current_size=%lu\n",payload_limit,result->result_size);
    if (result->result_size >= payload_limit) {
        femu_log("[AGG][APPEND] payload already full\n");
        return true;
    }
    remain = payload_limit - result->result_size;
    max_entries = remain / sizeof(diskann_neighbor_list_t);

    femu_log("[AGG][APPEND] remain=%lu max_entries=%lu\n",remain,max_entries);
    if (max_entries == 0) {
        femu_log("[AGG][APPEND] no remaining space\n");
        return true;
    }

    if ((uint64_t)list_cnt > max_entries) {
        femu_log("[AGG][APPEND] truncate list_cnt %u -> %lu\n",list_cnt,max_entries);
        list_cnt = (uint32_t)max_entries;
    }

    append_size = (uint64_t)list_cnt * sizeof(diskann_neighbor_list_t);
    new_size = result->result_size + append_size;
    femu_log("[AGG][APPEND] append_size=%lu new_size=%lu\n",append_size,new_size);
    femu_log("[AGG][APPEND] before realloc old_buf=%p\n",result->result_buf);
    new_buf = g_realloc(result->result_buf,new_size);
    femu_log("[AGG][APPEND] after realloc new_buf=%p\n",new_buf);
    if (!new_buf) {
        femu_err("[AGG][APPEND] realloc failed size=%lu\n",new_size);
        return false;
    }

    femu_log("[AGG][APPEND] memcpy dst=%p src=%p size=%lu\n",new_buf + result->result_size,lists,append_size);
    memcpy(new_buf + result->result_size,lists,append_size);
    femu_log("[AGG][APPEND] memcpy done\n");

    result->result_buf = new_buf;
    result->result_size = new_size;
    femu_log("[AGG][APPEND] leave success final_size=%lu\n", result->result_size);
    return true;
}
 
static inline bool visited_test_and_set(visited_set_t *v, uint64_t id)
{
    uint64_t byte;
    uint8_t bit;
    if (!v || !v->bits) {
        return true;
    }
    if (id >= MAX_DISKANN_NODES) {
        return true;
    }
    byte = id >> 3;
    bit = (uint8_t)(1u << (id & 7));
    if (v->bits[byte] & bit) {
        return true;
    }
    v->bits[byte] |= bit;
    return false;
}

static inline void visited_destroy(visited_set_t *v)
{
    if (v->bits) {
        free(v->bits);
    }
    v->bits = NULL;
}


 
bool femu_read_diskann_node( FemuCtrl *n, femu_query_ctx_t *qctx, uint64_t node_id, diskann_node_t **node_out)
{
    uint64_t stime;
    uint64_t lat = 0;
    diskann_node_t *node;
    if (!n || !n->ssd || !n->ssd->lftl || !node_out) {
        return false;
    }
    if (node_id >= MAX_DISKANN_NODES) {
        femu_err("Aggregation invalid node_id=%lu\n", node_id);
        return false;
    }
    stime = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);      
    node = g_malloc0(sizeof(diskann_node_t));            
    if (!node) {
        return false;
    }
                                                        
    if (ssd_read_nodeid_to_buf(n->ssd, n, node_id, node, sizeof(diskann_node_t), stime, &lat) != 0) {   
        g_free(node);
        return false;
    }

    if (qctx && qctx->read_cnt < MAX_QUERY_READS) {
        qctx->reads[qctx->read_cnt++] = (query_io_t) {
            .type = QUERY_IO_READ,
            .node_id_or_lba = node_id,
            .nlb = LEARNED_NODE_STRIDE_4K,
            .issue_ts = stime,
            .latency = lat
        };
    }
    if (node->neighbor_count > DISKANN_MAX_DEGREE) {
        femu_err("Aggregation corrupted node=%lu degree=%u max=%u\n", node_id,  node->neighbor_count, DISKANN_MAX_DEGREE );
        g_free(node);
        return false;
    }

    *node_out = node;
    return true;
}


void femu_dump_diskann_node(FemuCtrl *n, uint64_t node_id)
{
    diskann_node_t *node;
    if (!femu_read_diskann_node(n, NULL, node_id, &node)) {
        femu_log("dump node failed: node=%lu\n", node_id);
        return;
    }
    femu_log("=== DiskANN node %lu ===\n", node_id);
    femu_log("degree=%u\n", node->neighbor_count);
    uint32_t deg = MIN(node->neighbor_count, DISKANN_MAX_DEGREE);
    for (uint32_t i = 0; i < deg; i++) {
        femu_log("  nbr[%u]=%u\n", i, node->neighbors[i]);
    }
}

 
bool femu_expand_and_collect_neighbors( FemuCtrl *n, femu_query_ctx_t *qctx, uint64_t root_node_id, diskann_neighbor_list_t *out, uint32_t *out_cnt)
{
    diskann_node_t *root = NULL;
    if (!femu_read_diskann_node(n, qctx, root_node_id, &root)) {    // 
        return false;
    }
    femu_log("[AGG] root=%lu degree=%u\n", root_node_id, root->neighbor_count);

    uint32_t deg = root->neighbor_count;
    if (deg > DISKANN_MAX_DEGREE) {
        deg = DISKANN_MAX_DEGREE;
    }
    uint32_t collected = 0;
    for (uint32_t i = 0; i < deg; i++) {
        uint64_t logical_nid = root->neighbors[i];
        uint64_t ssd_nid = logical_to_ssd_nodeid(logical_nid);
        diskann_node_t *nbr = NULL;

        if (!femu_read_diskann_node(n, qctx, ssd_nid, &nbr)) {
            continue;
        }

        out[collected].node_id = (uint32_t)logical_nid;
        out[collected].degree = nbr->neighbor_count > DISKANN_MAX_DEGREE ? DISKANN_MAX_DEGREE : nbr->neighbor_count;

        memcpy(out[collected].neighbors, nbr->neighbors, out[collected].degree * sizeof(uint32_t));
        g_free(nbr);
        collected++;
        if (collected >= DISKANN_MAX_DEGREE) {
            break;
        }
    }
    g_free(root);
    *out_cnt = collected;
    return true;
}

 
static bool femu_dfs_recursive(FemuCtrl *n,femu_query_ctx_t *qctx,femu_query_result_t *result,uint64_t root_node_id,uint32_t hop,uint32_t max_hop)
{
    diskann_neighbor_list_t *lists = NULL;
    uint32_t list_cnt = 0;
    bool ret = false;
    if (hop > max_hop) {
        return true;
    }
    lists = g_malloc0(sizeof(diskann_neighbor_list_t) * DISKANN_MAX_DEGREE);
    if (!lists) {
        femu_err("[AGG] alloc lists failed\n");
        return false;
    }
    if (!femu_expand_and_collect_neighbors(n, qctx, root_node_id, lists, &list_cnt)) {
        femu_log("[AGG] expand failed root=%lu hop=%u\n", root_node_id, hop);
        goto out;
    }
    if (hop == 0 && list_cnt == 0) {
        goto out;
    }
    if (list_cnt > 0) {
        if (!result_append_lists(result, lists, list_cnt)) {
            goto out;
        }
    }
    qctx->num_hops++;
    for (uint32_t i = 0; i < list_cnt; i++) {
        for (uint32_t j = 0; j < lists[i].degree; j++) {
            uint64_t next_logical = lists[i].neighbors[j];
            uint64_t next_ssd = logical_to_ssd_nodeid(next_logical);
            if (visited_test_and_set(&qctx->visited, next_logical)) {
                continue;
            }
            if (!femu_dfs_recursive(n, qctx, result, next_ssd, hop + 1, max_hop)) {
                femu_log("[AGG] skip failed child root=%lu next=%lu hop=%u\n", root_node_id, next_ssd, hop + 1);
                continue;
            }
        }
    }
    ret = true;
out:
    if (lists) {
        g_free(lists);
    }

    return ret;
}

 
static bool femu_fill_query_result(femu_query_ctx_t *qctx, femu_query_result_t *result)
{
    uint64_t max_read_lat = 0;
    uint64_t sum_read_lat = 0;

    if (!qctx || !result) {
        return false;
    }

    // femu_log("[AGG_LAT] read_cnt=%u\n", qctx->read_cnt);

    for (uint32_t i = 0; i < qctx->read_cnt; i++) {
        uint64_t lat = qctx->reads[i].latency;

        sum_read_lat += lat;

        if (lat > max_read_lat) {
            max_read_lat = lat;
        }

        // femu_log("[AGG_LAT] i=%u node=%lu nlb=%u lat=%lu\n",
        //          i,
        //          qctx->reads[i].node_id_or_lba,
        //          qctx->reads[i].nlb,
        //          lat);
    }

    femu_log("[AGG_LAT] read_cnt=%u max=%lu sum=%lu\n",
             qctx->read_cnt,
             max_read_lat,
             sum_read_lat);

    result->total_read_lat = max_read_lat;
    result->total_write_lat = 0;
    result->total_lat = max_read_lat;
    result->read_cnt = qctx->read_cnt;
    result->write_cnt = 0;

    return true;
}

bool femu_execute_query_dfs(FemuCtrl *n, uint32_t max_hop, uint64_t root_node_id, femu_query_result_t *result)
{
    femu_query_ctx_t *qctx = NULL;
    bool ok = false;
    if (!n || !result) {
        return false;
    }
    qctx = g_malloc0(sizeof(*qctx));
    if (!qctx) {
        return false;
    }
    result->total_read_lat = 0;
    result->total_write_lat = 0;
    result->total_lat = 0;
    result->read_cnt = 0;
    result->write_cnt = 0;
    result->result_buf = NULL;
    result->result_size = 0;
    qctx->query_start_ts = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
    qctx->query_id = root_node_id;

    visited_init(&qctx->visited);
    if (!qctx->visited.bits) {
        g_free(qctx);
        return false;
    }

    visited_test_and_set(&qctx->visited, root_node_id-1);
    ok = femu_dfs_recursive(n,qctx,result,root_node_id,0,max_hop);
    if (!ok || qctx->num_hops == 0 || result->result_size == 0) {
        goto fail;
    }
    if (!femu_fill_query_result(qctx, result)) {
        goto fail;
    }
    visited_destroy(&qctx->visited);
    g_free(qctx);
    return true;
fail:
    if (result->result_buf) {
        g_free(result->result_buf);
        result->result_buf = NULL;
    }
    result->result_size = 0;
    visited_destroy(&qctx->visited);
    g_free(qctx);
    return false;
}

static bool result_append_prefetch_nodes(femu_query_result_t *result,diskann_prefetch_node_t *nodes,uint32_t node_cnt)
{
    uint64_t payload_limit;
    uint64_t remain;
    uint64_t max_entries;
    uint64_t append_size;
    uint64_t new_size;
    uint8_t *new_buf;

    if (!result || !nodes || node_cnt == 0) {
        return true;
    }

    payload_limit = result->result_capacity;
    if (payload_limit == 0 || result->result_size >= payload_limit) {
        return true;
    }

    remain = payload_limit - result->result_size;
    max_entries = remain / sizeof(diskann_prefetch_node_t);

    if (max_entries == 0) {
        return true;
    }

    if ((uint64_t)node_cnt > max_entries) {
        femu_log("[AGG][APPEND_PREFETCH] truncate node_cnt %u -> %lu\n",node_cnt, max_entries);
        node_cnt = (uint32_t)max_entries;
    }

    append_size = (uint64_t)node_cnt * sizeof(diskann_prefetch_node_t);
    new_size = result->result_size + append_size;
    new_buf = g_realloc(result->result_buf, new_size);
    if (!new_buf) {
        femu_err("[AGG][APPEND_PREFETCH] realloc failed size=%lu\n", new_size);
        return false;
    }
    memcpy(new_buf + result->result_size, nodes, append_size);
    result->result_buf = new_buf;
    result->result_size = new_size;
    return true;
}



bool femu_execute_query_batch(FemuCtrl *n,uint64_t query_start_ts, uint64_t *node_ids, uint32_t node_num, femu_query_result_t *result)
{
    femu_query_ctx_t *qctx = NULL;
    diskann_prefetch_node_t *prefetch_nodes = NULL;
    bool ok = false;
    if (!n || !node_ids || !result) {
        return false;
    }

    if (node_num == 0 || node_num > MAX_NODE_IDS) {
        femu_err("[AGG][BATCH] invalid node_num=%u\n", node_num);
        return false;
    }

    result->total_read_lat = 0;
    result->total_write_lat = 0;
    result->total_lat = 0;
    result->read_cnt = 0;
    result->write_cnt = 0;
    result->result_buf = NULL;
    result->result_size = 0;
    qctx = g_malloc0(sizeof(*qctx));
    if (!qctx) {
        return false;
    }

    qctx->query_start_ts = query_start_ts;
    qctx->query_id = node_ids[0];

    prefetch_nodes = g_malloc0(sizeof(diskann_prefetch_node_t) * node_num);
    if (!prefetch_nodes) {
        femu_err("[AGG][BATCH] alloc prefetch_nodes failed node_num=%u\n", node_num);
        goto out;
    }

    uint32_t node_cnt = 0;
    for (uint32_t i = 0; i < node_num; i++) {
        uint64_t node_id = node_ids[i];
        diskann_node_t *node = NULL;
        if (!femu_read_diskann_node(n, qctx, node_id, &node)) {
            femu_log("[AGG][BATCH] read node failed node=%lu\n", node_id);
            continue;
        }
        prefetch_nodes[node_cnt].node_id = (uint32_t)node_id;
        memcpy(prefetch_nodes[node_cnt].vector,node->vector,DISKANN_NODE_VEC_BYTES);

        if (node->neighbor_count > DISKANN_MAX_DEGREE) {
            prefetch_nodes[node_cnt].degree = DISKANN_MAX_DEGREE;
        } else {
            prefetch_nodes[node_cnt].degree = node->neighbor_count;
        }

        if (prefetch_nodes[node_cnt].degree > 0) {
            memcpy(prefetch_nodes[node_cnt].neighbors,node->neighbors,prefetch_nodes[node_cnt].degree * sizeof(uint32_t));
        }
        g_free(node);
        node_cnt++;
    }

    if (node_cnt == 0) {
        femu_log("[AGG][BATCH] no valid node collected\n");
        goto out;
    }

    if (!result_append_prefetch_nodes(result, prefetch_nodes, node_cnt)) {
        femu_err("[AGG][BATCH] append prefetch result failed\n");
        goto out;
    }

    if (!femu_fill_query_result(qctx, result)) {
        femu_err("[AGG][BATCH] fill result failed\n");
        goto out;
    }
    ok = true;
out:
    if (!ok && result->result_buf) {
        g_free(result->result_buf);
        result->result_buf = NULL;
        result->result_size = 0;
    }
    if (prefetch_nodes) {
        g_free(prefetch_nodes);
    }
    if (qctx) {
        // visited_destroy(&qctx->visited);
        g_free(qctx);
    }
    return ok;
}


bool femu_execute_aggregation_param(FemuCtrl *n,struct ParamEntry *cmd,void *result_buf,uint64_t result_buf_size,uint64_t *lat_out)
{
    agg_result_meta_t meta;
    femu_query_result_t result;
    bool ok = false;
    if (!n || !cmd || !result_buf || result_buf_size < sizeof(meta)) {
        return false;
    }

    memset(&meta, 0, sizeof(meta));
    memset(&result, 0, sizeof(result));
    memset(result_buf, 0, result_buf_size);
    result.result_capacity = result_buf_size - sizeof(agg_result_meta_t);

    if (cmd->node_num == 0 || cmd->node_num > MAX_NODE_IDS) {
        femu_err("[AGGR_PARAM] invalid node_num=%u\n", cmd->node_num);
        goto out;
    }

    if (cmd->flag == 0) {
        if (cmd->node_num != 1) {
            femu_err("[AGGR_PARAM] DFS only supports node_num=1, got %u\n",cmd->node_num);
            goto out;
        }
        ok = femu_execute_query_dfs(n, cmd->hop, cmd->node_id[0], &result);
    } else if (cmd->flag == 1) {
        ok = femu_execute_query_batch(n,qemu_clock_get_ns(QEMU_CLOCK_REALTIME), cmd->node_id, cmd->node_num, &result);
    } else {
        femu_err("[AGGR_PARAM] invalid flag=%u\n", cmd->flag);
        goto out;
    }

out:
    meta.magic = AGG_RES_MAGIC;
    meta.status = ok ? 0 : 1;
    meta.result_size = ok ? result.result_size : 0;
    meta.total_latency = ok ? result.total_lat : 0;
    meta.read_latency = ok ? result.total_read_lat : 0;
    meta.write_latency = ok ? result.total_write_lat : 0;
    meta.read_cnt = ok ? result.read_cnt : 0;
    meta.write_cnt = ok ? result.write_cnt : 0;

    if (ok && sizeof(meta) + result.result_size > result_buf_size) {
        femu_err("[AGGR_PARAM] result too large: result=%lu buf=%lu\n", result.result_size, result_buf_size);
        meta.status = 1;
        meta.result_size = 0;
        meta.total_latency = 0;
        meta.read_latency = 0;
        meta.write_latency = 0;
        meta.read_cnt = 0;
        meta.write_cnt = 0;
        ok = false;
    }

    memcpy(result_buf, &meta, sizeof(meta));

    if (ok && result.result_buf && result.result_size > 0) {
        memcpy((uint8_t *)result_buf + sizeof(meta), result.result_buf, result.result_size);
    }
    if (lat_out) {
        *lat_out = meta.total_latency;
    }
    if (result.result_buf) {
        g_free(result.result_buf);
    }
    return ok;
}

uint16_t nvme_write_result_prp(FemuCtrl *n,NvmeRequest *req,uint64_t prp1,uint64_t prp2,void *buf,uint64_t len)
{
    QEMUSGList qsg;
    QEMUIOVector iov;
    if (!n || !req || !buf || len == 0 || !prp1) {
        return NVME_INVALID_FIELD | NVME_DNR;
    }
    memset(&qsg, 0, sizeof(qsg));
    memset(&iov, 0, sizeof(iov));
    if (nvme_map_prp(&qsg, &iov, prp1, prp2, len, n)) {
        return NVME_INVALID_FIELD | NVME_DNR;
    }
    uint64_t copied = 0;
    for (int idx = 0; idx < qsg.nsg; idx++) {
        uint64_t cur_len = qsg.sg[idx].len;
        if (copied + cur_len > len) {
            cur_len = len - copied;
        }
        if (dma_memory_rw(qsg.as,qsg.sg[idx].base,(uint8_t *)buf + copied,cur_len,DMA_DIRECTION_FROM_DEVICE,MEMTXATTRS_UNSPECIFIED)) {
            qemu_sglist_destroy(&qsg);
            return NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        }
        copied += cur_len;
        if (copied >= len) {
            break;
        }
    }
    qemu_sglist_destroy(&qsg);
    return copied == len ? NVME_SUCCESS : (NVME_INVALID_FIELD | NVME_DNR);
}