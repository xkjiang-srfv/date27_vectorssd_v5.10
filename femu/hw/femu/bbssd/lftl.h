#ifndef __FEMU_LEARNED_FTL_H__
#define __FEMU_LEARNED_FTL_H__
#include "../nvme.h"
#include "./raw_region.h"
#include "diskann_format.h"


#define LBA_BYTES                  512ULL                                

#define FLASH_PAGE_BYTES           4096ULL
#define FLASH_PAGE_SHIFT           12u
#define SECS_PER_FLASH_PAGE        (FLASH_PAGE_BYTES / LBA_BYTES)

#define LEARNED_NODE_BYTES         DISKANN_NODE_BYTES

#define LEARNED_NODE_STRIDE_PAGES \
    ((uint32_t)((LEARNED_NODE_BYTES + FLASH_PAGE_BYTES - 1) / FLASH_PAGE_BYTES))

#define LEARNED_NODES_PER_PAGE \
    ((LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) ? \
    (FLASH_PAGE_BYTES / LEARNED_NODE_BYTES) : 1)

#define LEARNED_NODE_SLOT_BYTES \
    ((LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) ? \
    LEARNED_NODE_BYTES : FLASH_PAGE_BYTES)

#define LEARNED_NODE_PAGE_WASTE \
    ((LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) ? \
    (FLASH_PAGE_BYTES % LEARNED_NODE_BYTES) : 0)

#define LEARNED_NODE_STRIDE_4K     LEARNED_NODE_STRIDE_PAGES


#ifndef MAX_DISKANN_NODES
#define MAX_DISKANN_NODES           ((uint64_t)0x8000000ULL)
#endif

#define LFTL_INVALID_U32            ((uint32_t)0xFFFFFFFFu)             // 
#define LFTL_INVALID_U64            ((uint64_t)0xFFFFFFFFFFFFFFFFull)    

typedef struct WorkerArg {
    FemuCtrl *n;
    int worker_id;
} WorkerArg;


 

struct learned_ftl {
    struct ssdparams *spp;
    uint64_t map_latency_ns;
    uint64_t map_count;
    int initialized;

    bool pending_partial_page;       
    uint64_t pending_last_nodeid;    
};


 

enum {
    NAND_READ = 0,
    NAND_WRITE = 1,
    NAND_ERASE = 2,
    NAND_READ_LATENCY = 80000,
    NAND_PROG_LATENCY = 800000,
    NAND_ERASE_LATENCY = 2000000,
};

enum {
    USER_IO = 0,
    GC_IO = 1,
};

enum {
    SEC_FREE = 0,
    SEC_INVALID = 1,
    SEC_VALID = 2,
    PG_FREE = 0,
    PG_INVALID = 1,
    PG_VALID = 2
};

enum {
    FEMU_ENABLE_GC_DELAY = 1,
    FEMU_DISABLE_GC_DELAY = 2,
    FEMU_ENABLE_DELAY_EMU = 3,
    FEMU_DISABLE_DELAY_EMU = 4,
    FEMU_RESET_ACCT = 5,
    FEMU_ENABLE_LOG = 6,
    FEMU_DISABLE_LOG = 7,
};

#define BLK_BITS    (16)
#define PG_BITS     (16)
#define SEC_BITS    (8)
#define PL_BITS     (8)
#define LUN_BITS    (8)
#define CH_BITS     (7)

struct ppa {
    union {
        struct {
            uint64_t blk : BLK_BITS;
            uint64_t pg  : PG_BITS;
            uint64_t sec : SEC_BITS;
            uint64_t pl  : PL_BITS;
            uint64_t lun : LUN_BITS;
            uint64_t ch  : CH_BITS;
            uint64_t rsv : 1;
        } g;
        uint64_t ppa;
    };
};

typedef int nand_sec_status_t;

struct nand_page {
    nand_sec_status_t *sec;
    int nsecs;
    int status;
};

struct nand_block {
    struct nand_page *pg;
    int npgs;
    int ipc;
    int vpc;
    int erase_cnt;
    int wp;
};

struct nand_plane {
    struct nand_block *blk;
    int nblks;
};

struct nand_cmd {
    int type;
    int cmd;
    int64_t stime;
};

struct nand_lun {
    struct nand_plane *pl;
    int npls;
    uint64_t next_lun_avail_time;
    bool busy;
    uint64_t gc_endtime;
};

struct ssd_channel {
    struct nand_lun *lun;
    int nluns;
    uint64_t next_ch_avail_time;
    bool busy;
    uint64_t gc_endtime;
};

struct ssdparams {
    int secsz;
    int secs_per_pg;
    int pgs_per_blk;
    int blks_per_pl;
    int pls_per_lun;
    int luns_per_ch;
    int nchs;
    int pg_rd_lat;
    int pg_wr_lat;
    int blk_er_lat;
    int ch_xfer_lat;
    double gc_thres_pcent;
    int gc_thres_lines;
    double gc_thres_pcent_high;
    int gc_thres_lines_high;
    bool enable_gc_delay;
    int secs_per_blk;
    int secs_per_pl;
    int secs_per_lun;
    int secs_per_ch;
    int tt_secs;
    int pgs_per_pl;
    int pgs_per_lun;
    int pgs_per_ch;
    int tt_pgs;
    int blks_per_lun;
    int blks_per_ch;
    int tt_blks;
    int secs_per_line;
    int pgs_per_line;
    int blks_per_line;
    int tt_lines;
    int pls_per_ch;
    int tt_pls;
    int tt_luns;
};

struct ssd {
    char *ssdname;
    struct ssdparams sp;
    struct ssd_channel *ch;
    struct learned_ftl *lftl;
    struct rte_ring **to_ftl;
    struct rte_ring **to_poller;
     
    QemuThread *raw_threads;
    struct rte_ring **to_raw;
     
    bool *dataplane_started_ptr;
    QemuThread lftl_thread;
};



void ssd_init(FemuCtrl *n);



 

struct learned_ftl *learned_ftl_init(struct ssdparams *spp);
void learned_ftl_free(struct learned_ftl *lftl);

uint64_t ssd_read_nodeid_internal(struct ssd *ssd, uint64_t nodeid, uint32_t nblks, uint64_t stime);
uint64_t ssd_read_nodeid(struct ssd *ssd, NvmeRequest *req);
uint64_t ssd_write_nodeid(struct ssd *ssd, NvmeRequest *req);

int learned_ftl_get_page_ppa(struct learned_ftl *lftl, uint64_t nodeid, uint32_t page_offset, struct ppa *ppa);

static inline bool learned_node_req_valid(struct ssd *ssd, uint64_t nodeid, uint32_t nblks)
{
    (void)ssd;

    if (LEARNED_NODES_PER_PAGE == 0) {
        femu_err("invalid layout: node_bytes=%lu flash_page_bytes=%lu\n", (uint64_t)LEARNED_NODE_BYTES, (uint64_t)FLASH_PAGE_BYTES);
        return false;
    }

    if (nodeid >= MAX_DISKANN_NODES) {
        femu_err("invalid nodeid=%lu\n", nodeid);
        return false;
    }

    if (nblks == 0 || nblks > LEARNED_NODE_STRIDE_PAGES) {
        femu_err("LFTL request too large: nblks=%u stride_pages=%u\n", nblks, LEARNED_NODE_STRIDE_PAGES);
        return false;
    }

    return true;
}

 
static inline bool valid_ppa(struct ssd *ssd, struct ppa *ppa)
{
    struct ssdparams *spp = &ssd->sp;
    int ch = ppa->g.ch;
    int lun = ppa->g.lun;
    int pl = ppa->g.pl;
    int blk = ppa->g.blk;
    int pg = ppa->g.pg;
    int sec = ppa->g.sec;
    if (ch >= 0 && ch < spp->nchs && lun >= 0 && lun < spp->luns_per_ch && pl >= 0 && pl < spp->pls_per_lun && blk >= 0 && blk < spp->blks_per_pl && pg >= 0 && pg < spp->pgs_per_blk && sec >= 0 && sec < spp->secs_per_pg) {
        return true;
    }
    return false;
}


 
static inline uint32_t lftl_global_parallel_units(struct ssdparams *spp)
{
    return (uint32_t)(spp->nchs * spp->luns_per_ch);
}

 
 
 
static inline uint64_t lftl_base_global_page_for_node(uint64_t node_id)
{
    if (node_id == 0) {
        return 0;
    }

    if (LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) {
        return 1 + (node_id - 1) / LEARNED_NODES_PER_PAGE;    
    } else {
        return 1 + (node_id - 1) * LEARNED_NODE_STRIDE_PAGES;
    }
}

static inline uint32_t lftl_node_slot_id(uint64_t node_id)
{
    if (node_id == 0) {
        return 0;
    }

    if (LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) {
        return (uint32_t)((node_id - 1) % LEARNED_NODES_PER_PAGE);
    } else {
        return 0;
    }
}

static inline uint64_t lftl_node_in_page_offset(uint64_t node_id)
{
    if (LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) {
        return (uint64_t)lftl_node_slot_id(node_id) * LEARNED_NODE_SLOT_BYTES;
    } else {
        return 0;
    }
}

 
static inline uint64_t lftl_node_byte_offset(uint64_t node_id)
{
    return lftl_base_global_page_for_node(node_id) * FLASH_PAGE_BYTES + lftl_node_in_page_offset(node_id);
}

 
static inline bool lftl_node_layout_valid(void)
{
    if (LEARNED_NODE_BYTES == 0 || FLASH_PAGE_BYTES == 0) {
        return false;
    }

    if (LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) {
        return LEARNED_NODES_PER_PAGE > 0 &&
               LEARNED_NODE_BYTES * LEARNED_NODES_PER_PAGE <= FLASH_PAGE_BYTES;
    } else {
        return LEARNED_NODE_STRIDE_PAGES > 1;
    }
}

 
int ssd_read_nodeid_to_buf(struct ssd *ssd, FemuCtrl *n, uint64_t nodeid, void *buf, uint32_t size, uint64_t stime, uint64_t *lat_out);



#ifdef FEMU_DEBUG_FTL
#define ftl_debug(fmt, ...) \
    do { printf("[FEMU] FTL-Dbg: " fmt, ## __VA_ARGS__); } while (0)
#else
#define ftl_debug(fmt, ...) \
    do { } while (0)
#endif

#define ftl_err(fmt, ...) \
    do { fprintf(stderr, "[FEMU] FTL-Err: " fmt, ## __VA_ARGS__); } while (0)

#define ftl_log(fmt, ...) \
    do { printf("[FEMU] FTL-Log: " fmt, ## __VA_ARGS__); } while (0)

#ifdef FEMU_DEBUG_FTL
#define ftl_assert(expression) assert(expression)
#else
#define ftl_assert(expression)
#endif



#endif /* __FEMU_LEARNED_FTL_H__ */