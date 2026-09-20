#include "lftl.h"
#include <math.h>
#include <string.h>
#include "aggregation.h"
#include "./diskann_format.h"

static void *raw_thread(void *arg);

 
 
static inline struct ssd_channel *get_ch(struct ssd *ssd, struct ppa *ppa)
{
    return &(ssd->ch[ppa->g.ch]);            
}

 
static inline struct nand_lun *get_lun(struct ssd *ssd, struct ppa *ppa)
{
    struct ssd_channel *ch = get_ch(ssd, ppa);   
    return &(ch->lun[ppa->g.lun]);               
}

 
static void check_params(struct ssdparams *spp)
{
    (void)spp;                       
}

 
static void ssd_init_params(struct ssdparams *spp, FemuCtrl *n)
{
    spp->secsz = n->bb_params.secsz;
    spp->secs_per_pg = n->bb_params.secs_per_pg;
    spp->pgs_per_blk = n->bb_params.pgs_per_blk;
    spp->blks_per_pl = n->bb_params.blks_per_pl;
    spp->pls_per_lun = n->bb_params.pls_per_lun;
    spp->luns_per_ch = n->bb_params.luns_per_ch;
    spp->nchs = n->bb_params.nchs;

    spp->pg_rd_lat = n->bb_params.pg_rd_lat;
    spp->pg_wr_lat = n->bb_params.pg_wr_lat;
    spp->blk_er_lat = n->bb_params.blk_er_lat;
    spp->ch_xfer_lat = n->bb_params.ch_xfer_lat;

    /* calculated values */
    spp->secs_per_blk = spp->secs_per_pg * spp->pgs_per_blk;
    spp->secs_per_pl = spp->secs_per_blk * spp->blks_per_pl;
    spp->secs_per_lun = spp->secs_per_pl * spp->pls_per_lun;
    spp->secs_per_ch = spp->secs_per_lun * spp->luns_per_ch;
    spp->tt_secs = spp->secs_per_ch * spp->nchs;

    spp->pgs_per_pl = spp->pgs_per_blk * spp->blks_per_pl;
    spp->pgs_per_lun = spp->pgs_per_pl * spp->pls_per_lun;
    spp->pgs_per_ch = spp->pgs_per_lun * spp->luns_per_ch;
    spp->tt_pgs = spp->pgs_per_ch * spp->nchs;

    spp->blks_per_lun = spp->blks_per_pl * spp->pls_per_lun;
    spp->blks_per_ch = spp->blks_per_lun * spp->luns_per_ch;
    spp->tt_blks = spp->blks_per_ch * spp->nchs;

    spp->pls_per_ch = spp->pls_per_lun * spp->luns_per_ch;
    spp->tt_pls = spp->pls_per_ch * spp->nchs;

    spp->tt_luns = spp->luns_per_ch * spp->nchs;

    spp->blks_per_line = spp->tt_luns;
    spp->pgs_per_line = spp->blks_per_line * spp->pgs_per_blk;
    spp->secs_per_line = spp->pgs_per_line * spp->secs_per_pg;
    spp->tt_lines = spp->blks_per_lun;

    spp->gc_thres_pcent = n->bb_params.gc_thres_pcent / 100.0;
    spp->gc_thres_lines = (int)((1 - spp->gc_thres_pcent) * spp->tt_lines);
    spp->gc_thres_pcent_high = n->bb_params.gc_thres_pcent_high / 100.0;
    spp->gc_thres_lines_high = (int)((1 - spp->gc_thres_pcent_high) * spp->tt_lines);
    spp->enable_gc_delay = true;

    check_params(spp);
}

 
static void ssd_init_nand_page(struct nand_page *pg, struct ssdparams *spp)
{
    pg->nsecs = spp->secs_per_pg;                                    
    pg->sec = g_malloc0(sizeof(nand_sec_status_t) * pg->nsecs);      
    for (int i = 0; i < pg->nsecs; i++) {                            
        pg->sec[i] = SEC_FREE;                                       
    }
    pg->status = PG_FREE;                                            
}

 
static void ssd_init_nand_blk(struct nand_block *blk, struct ssdparams *spp)
{
    blk->npgs = spp->pgs_per_blk;                                    
    blk->pg = g_malloc0(sizeof(struct nand_page) * blk->npgs);       
    for (int i = 0; i < blk->npgs; i++) {                            
        ssd_init_nand_page(&blk->pg[i], spp);                        
    }
    blk->ipc = 0;                    
    blk->vpc = 0;                    
    blk->erase_cnt = 0;              
    blk->wp = 0;                     
}

 
static void ssd_init_nand_plane(struct nand_plane *pl, struct ssdparams *spp)
{
    pl->nblks = spp->blks_per_pl;                                
    pl->blk = g_malloc0(sizeof(struct nand_block) * pl->nblks);  
    for (int i = 0; i < pl->nblks; i++) {                        
        ssd_init_nand_blk(&pl->blk[i], spp);                     
    }
}

 
static void ssd_init_nand_lun(struct nand_lun *lun, struct ssdparams *spp)
{
    lun->npls = spp->pls_per_lun;                                    
    lun->pl = g_malloc0(sizeof(struct nand_plane) * lun->npls);      
    for (int i = 0; i < lun->npls; i++) {                            
        ssd_init_nand_plane(&lun->pl[i], spp);                       
    }
    lun->next_lun_avail_time = 0;                                    
    lun->busy = false;                                               
    lun->gc_endtime = 0;                                             
}

 
static void ssd_init_ch(struct ssd_channel *ch, struct ssdparams *spp)
{
    ch->nluns = spp->luns_per_ch;                                
    ch->lun = g_malloc0(sizeof(struct nand_lun) * ch->nluns);    
    for (int i = 0; i < ch->nluns; i++) {                        
        ssd_init_nand_lun(&ch->lun[i], spp);                     
    }
    ch->next_ch_avail_time = 0;                                  
    ch->busy = 0;                                                
    ch->gc_endtime = 0;                                          
}

 
static void lftl_mark_physical_page_valid_with_ssd(struct ssd *ssd, const struct ppa *ppa)
{
    struct nand_lun *lun;
    struct nand_plane *pl;
    struct nand_block *blk;
    struct nand_page *pg;
    if (!ssd || !ppa) {
        return;
    }

    lun = get_lun(ssd, (struct ppa *)ppa);
    pl = &lun->pl[ppa->g.pl];
    blk = &pl->blk[ppa->g.blk];
    pg = &blk->pg[ppa->g.pg];

    if (pg->status != PG_VALID) {
        pg->status = PG_VALID;
        for (int s = 0; s < pg->nsecs; s++) {
            pg->sec[s] = SEC_VALID;
        }
        blk->vpc++;
        if (blk->wp <= (int)ppa->g.pg) {
            blk->wp = (int)ppa->g.pg + 1;
        }
    }
}

struct learned_ftl *learned_ftl_init(struct ssdparams *spp)
{
    struct learned_ftl *lftl = g_malloc0(sizeof(*lftl));
    if (!lftl) {
        return NULL;
    }

    lftl->spp = spp;
    lftl->initialized = 1;
    lftl->pending_partial_page = false;
    lftl->pending_last_nodeid = 0;
    return lftl;
}

 
static void lftl_default_page_layout(struct learned_ftl *lftl,uint64_t node_id,uint32_t page_offset,uint32_t *channel_out,uint32_t *die_out, uint64_t *vpage_out)
{
    struct ssdparams *spp = lftl->spp;
    uint64_t base_global_page = lftl_base_global_page_for_node(node_id);     
    uint64_t global_page_idx = base_global_page + (uint64_t)page_offset;     
    uint32_t parallel_units = lftl_global_parallel_units(spp);               
    uint32_t stripe_unit = (uint32_t)(global_page_idx % parallel_units);     
    uint32_t ch = stripe_unit % (uint32_t)spp->nchs;                         
    uint32_t die = stripe_unit / (uint32_t)spp->nchs;
    uint64_t stripe_round = global_page_idx / parallel_units;                

    if (channel_out) {
        *channel_out = ch;
    }
    if (die_out) {
        *die_out = die;
    }
    if (vpage_out) {
        *vpage_out = stripe_round;
    }
}

 
 
int learned_ftl_get_page_ppa(struct learned_ftl *lftl, uint64_t nodeid, uint32_t page_offset, struct ppa *ppa_out)
{
    struct ssdparams *spp;
    uint32_t ch, die, pg_in_blk;
    uint64_t vpage, pblk;

    if (!lftl || !ppa_out) {
        return -1;
    }
    if (page_offset >= LEARNED_NODE_STRIDE_PAGES) {
        femu_err("LFTL page_offset overflow: node=%" PRIu64 " off=%u stride_pages=%u\n", nodeid, page_offset, LEARNED_NODE_STRIDE_PAGES);
        return -1;
    }

    spp = lftl->spp;
    if (!spp || spp->pgs_per_blk <= 0 || spp->blks_per_pl <= 0 || spp->nchs <= 0 || spp->luns_per_ch <= 0) {
        return -1;
    }

    lftl_default_page_layout(lftl, nodeid, page_offset, &ch, &die, &vpage);
    pblk = vpage / (uint64_t)spp->pgs_per_blk;
    pg_in_blk = (uint32_t)(vpage % (uint64_t)spp->pgs_per_blk);

    if (pblk >= (uint64_t)spp->blks_per_pl) {
        femu_err("LFTL block out of range: node=%" PRIu64 " ch=%u die=%u vpage=%" PRIu64 " pblk=%" PRIu64 "\n", nodeid, ch, die, vpage, pblk);
        return -1;
    }

    memset(ppa_out, 0, sizeof(*ppa_out));
    ppa_out->g.ch  = ch;
    ppa_out->g.lun = die;
    ppa_out->g.pl  = 0;
    ppa_out->g.blk = (uint32_t)pblk;
    ppa_out->g.pg  = pg_in_blk;
    ppa_out->g.sec = 0;
    return 0;
}

 
static uint64_t ssd_advance_status(struct ssd *ssd, struct ppa *ppa, struct nand_cmd *ncmd)
{
    int c = ncmd->cmd;           
    uint64_t cmd_stime = (ncmd->stime == 0) ? qemu_clock_get_ns(QEMU_CLOCK_REALTIME) : ncmd->stime;      
    uint64_t nand_stime;         
    struct ssdparams *spp = &ssd->sp;                
    struct nand_lun *lun = get_lun(ssd, ppa);        
    uint64_t lat = 0;                                
    switch (c) {
    case NAND_READ:                                  
        nand_stime = (lun->next_lun_avail_time < cmd_stime) ? cmd_stime : lun->next_lun_avail_time;      
        lun->next_lun_avail_time = nand_stime + spp->pg_rd_lat;                                          
        lat = lun->next_lun_avail_time - cmd_stime;                                                      
        break;
    case NAND_WRITE:
        nand_stime = (lun->next_lun_avail_time < cmd_stime) ? cmd_stime : lun->next_lun_avail_time;     // 
        lun->next_lun_avail_time = nand_stime + spp->pg_wr_lat;
        lat = lun->next_lun_avail_time - cmd_stime;
        break;
    case NAND_ERASE:
        nand_stime = (lun->next_lun_avail_time < cmd_stime) ? cmd_stime : lun->next_lun_avail_time;
        lun->next_lun_avail_time = nand_stime + spp->blk_er_lat;
        lat = lun->next_lun_avail_time - cmd_stime;
        break;
    default:
        ftl_err("Unsupported NAND command: 0x%x\n", c);
        break;
    }
    return lat;
}

 
uint64_t ssd_read_nodeid(struct ssd *ssd, NvmeRequest *req)
{   
    uint64_t nodeid = req->slba;
    uint32_t nblks = (uint32_t)req->nlb;
    uint64_t maxlat = 0;
    if (!learned_node_req_valid(ssd, nodeid, nblks)) {
        return 0;
    }
     
    for (uint32_t i = 0; i < nblks; i++) {
        struct ppa ppa;
        if (learned_ftl_get_page_ppa(ssd->lftl, nodeid, i, &ppa) != 0) {              
            femu_err("LFTL read invalid PPA: node=%" PRIu64 " off=%u\n",nodeid,i);
            continue;
        }
        if (!valid_ppa(ssd, &ppa)) {
            femu_err("LFTL read invalid PPA(after assemble): node=%" PRIu64 " off=%u\n",nodeid,i);
            continue;
        }
         
        struct nand_cmd srd;
        memset(&srd, 0, sizeof(srd));
        srd.type = USER_IO;
        srd.cmd = NAND_READ;
        srd.stime = req->stime;
        uint64_t lat = ssd_advance_status(ssd, &ppa, &srd);
        if (lat > maxlat) {
            maxlat = lat;
        }
    }
    ssd->lftl->map_count++;
    return maxlat;
}

 
uint64_t ssd_read_nodeid_internal(struct ssd *ssd, uint64_t nodeid, uint32_t nblks, uint64_t stime)
{
    uint64_t maxlat = 0;
    if (!learned_node_req_valid(ssd, nodeid, nblks)) {
        return 0;
    }

    for (uint32_t i = 0; i < nblks; i++) {
        struct ppa ppa;
        struct nand_cmd srd;
        uint64_t lat;

        if (learned_ftl_get_page_ppa(ssd->lftl, nodeid, i, &ppa) != 0) {
            femu_err("LFTL read invalid PPA: node=%" PRIu64 " off=%u\n",nodeid,i);
            continue;
        }
        if (!valid_ppa(ssd, &ppa)) {
            femu_err("LFTL read invalid PPA(after assemble): node=%" PRIu64 " off=%u\n",nodeid,i);
            continue;
        }
        memset(&srd, 0, sizeof(srd));
        srd.type = USER_IO;
        srd.cmd = NAND_READ;
        srd.stime = stime;
        lat = ssd_advance_status(ssd, &ppa, &srd);
        if (lat > maxlat) {
            maxlat = lat;
        }
    }
    ssd->lftl->map_count++;
    return maxlat;
}

int ssd_read_nodeid_to_buf(struct ssd *ssd, FemuCtrl *n, uint64_t nodeid, void *buf, uint32_t size, uint64_t stime, uint64_t *lat_out)
{
    uint64_t lat;
    uint64_t off;
    if (nodeid == 0) {
        return -1;
    }
    if (size > LEARNED_NODE_BYTES) {
        size = LEARNED_NODE_BYTES;
    }
    lat = ssd_read_nodeid_internal(ssd, nodeid, LEARNED_NODE_STRIDE_PAGES, stime);
    if (lat_out) {
        *lat_out = lat;
    }
    off = lftl_node_byte_offset(nodeid);
    memcpy(buf,(uint8_t *)n->mbe->logical_space + off,size);
    return 0;
}


static uint64_t lftl_flush_partial_page(struct ssd *ssd, NvmeRequest *req)
{
    struct learned_ftl *lftl = ssd->lftl;

    if (!lftl->pending_partial_page) {
        return 0;
    }

    uint64_t nodeid = lftl->pending_last_nodeid;
    struct ppa ppa;
    struct nand_cmd swr;

    if (learned_ftl_get_page_ppa(lftl, nodeid, 0, &ppa) != 0) {
        femu_err("LFTL partial-page flush invalid PPA: node=%" PRIu64 "\n", nodeid);
        return 0;
    }

    if (!valid_ppa(ssd, &ppa)) {
        femu_err("LFTL partial-page flush invalid PPA(after assemble): node=%" PRIu64 "\n", nodeid);
        return 0;
    }

    memset(&swr, 0, sizeof(swr));
    swr.type = USER_IO;
    swr.cmd = NAND_WRITE;
    swr.stime = req->stime;

    uint64_t lat = ssd_advance_status(ssd, &ppa, &swr);

    lftl_mark_physical_page_valid_with_ssd(ssd, &ppa);

    femu_log("LFTL flush partial page: last_node=%" PRIu64 ", slot=%u\n", nodeid, lftl_node_slot_id(nodeid));

    lftl->pending_partial_page = false;

    return lat;
}


 
static uint64_t ssd_write_nodeid_default(struct ssd *ssd, NvmeRequest *req)
{
    uint64_t nodeid = req->slba;
    uint32_t nblks = (uint32_t)req->nlb;
    uint64_t maxlat = 0;
    struct learned_ftl *lftl = ssd->lftl;

    /*
     * Small-node layout:
     * Multiple DiskANN nodes share one physical flash page.
     */
    if (LEARNED_NODE_BYTES <= FLASH_PAGE_BYTES) {
        uint32_t slot_id = lftl_node_slot_id(nodeid);
        bool page_full = (slot_id == LEARNED_NODES_PER_PAGE - 1);

        /*
         * This page now contains at least one uncommitted node.
         */
        lftl->pending_partial_page = true;
        lftl->pending_last_nodeid = nodeid;

        /*
         * Page is not full yet.
         * Data is already in logical_space, so do not charge NAND timing.
         */
        if (!page_full) {
            lftl->map_count++;
            return 0;
        }

        /*
         * Last slot arrived: program the complete physical page once.
         */
        struct ppa ppa;
        struct nand_cmd swr;

        if (learned_ftl_get_page_ppa(lftl, nodeid, 0, &ppa) != 0) {
            femu_err("LFTL write invalid PPA: node=%" PRIu64 "\n", nodeid);
            return 0;
        }

        if (!valid_ppa(ssd, &ppa)) {
            femu_err("LFTL write invalid PPA(after assemble): node=%" PRIu64 "\n", nodeid);
            return 0;
        }

        memset(&swr, 0, sizeof(swr));
        swr.type = USER_IO;
        swr.cmd = NAND_WRITE;
        swr.stime = req->stime;

        maxlat = ssd_advance_status(ssd, &ppa, &swr);

        lftl_mark_physical_page_valid_with_ssd(ssd, &ppa);

        /*
         * This page has been fully programmed.
         * There is no pending partial page anymore.
         */
        lftl->pending_partial_page = false;

        lftl->map_count++;
        return maxlat;
    }

    /*
     * Large-node layout:
     * One node spans one or more flash pages.
     */
    for (uint32_t i = 0; i < nblks; i++) {
        struct ppa ppa;
        struct nand_cmd swr;

        if (learned_ftl_get_page_ppa(lftl, nodeid, i, &ppa) != 0) {
            femu_err("LFTL write invalid PPA: node=%" PRIu64 " off=%u\n", nodeid, i);
            continue;
        }

        if (!valid_ppa(ssd, &ppa)) {
            femu_err("LFTL write invalid PPA(after assemble): node=%" PRIu64 " off=%u\n", nodeid, i);
            continue;
        }

        memset(&swr, 0, sizeof(swr));
        swr.type = USER_IO;
        swr.cmd = NAND_WRITE;
        swr.stime = req->stime;

        uint64_t lat = ssd_advance_status(ssd, &ppa, &swr);

        if (lat > maxlat) {
            maxlat = lat;
        }

        lftl_mark_physical_page_valid_with_ssd(ssd, &ppa);
    }

    lftl->map_count++;
    return maxlat;
}

uint64_t ssd_write_nodeid(struct ssd *ssd, NvmeRequest *req)
{
    uint64_t nodeid = req->slba;
    uint32_t nblks = (uint32_t)req->nlb;    
    if (nodeid == 0) {
        return lftl_flush_partial_page(ssd, req);
    }
    if (!learned_node_req_valid(ssd, nodeid, nblks)) {           
        return 0;
    }

    return ssd_write_nodeid_default(ssd, req);
}

 
static void *lftl_thread(void *arg)
{
    FemuCtrl *n = (FemuCtrl *)arg;
    struct ssd *ssd = n->ssd;
    NvmeRequest *req = NULL;
    uint64_t lat = 0;
    int rc;
    int i;

    while (!*(ssd->dataplane_started_ptr)) {
        usleep(100000);
    }

    ssd->to_ftl = n->to_ftl;
    ssd->to_poller = n->to_poller;
    while (1) {
        for (i = 1; i <= n->nr_pollers; i++) {
            if (!ssd->to_ftl[i] || !femu_ring_count(ssd->to_ftl[i])) {
                continue;
            }
            rc = femu_ring_dequeue(ssd->to_ftl[i], (void *)&req, 1);
            if (rc != 1) {
                printf("FEMU: FTL to_ftl dequeue failed\n");
            }
            ftl_assert(req);
            switch (req->cmd.opcode) {
            case NVME_CMD_NODEID_READ:
                lat = ssd_read_nodeid(ssd, req);
                break;
            case NVME_CMD_NODEID_WRITE:
                lat = ssd_write_nodeid(ssd, req);
                break;
            default:
                break;
            }
            req->reqlat = lat;
            req->expire_time += lat;
            rc = femu_ring_enqueue(ssd->to_poller[i], (void *)&req, 1);
            if (rc != 1) {
                ftl_err("FTL to_poller enqueue failed\n");
            }
        }
    }
    return NULL;
}
 


static void *raw_thread(void *arg)
{
    WorkerArg *warg = (WorkerArg *)arg;
    FemuCtrl *n = warg->n;
    int i = warg->worker_id;
    struct ssd *ssd = n->ssd;
    NvmeRequest *req = NULL;
    int rc;
    femu_log("[] RAW worker start id=%d nr_pollers=%d\n", i, n->nr_pollers);

    while (!*(ssd->dataplane_started_ptr)) {
        usleep(100000);
    }

    ssd->to_raw = n->to_raw;
    ssd->to_poller = n->to_poller;

    while (1) {
        req = NULL;
        if (!ssd->to_raw[i] || !femu_ring_count(ssd->to_raw[i])) {
            usleep(1);
            continue;
        }
        rc = femu_ring_dequeue(ssd->to_raw[i], (void *)&req, 1);
        if (rc != 1 || req == NULL) {
            femu_err("RAW to_raw dequeue failed, ret=%d\n", rc);
            continue;
        }
        ftl_assert(req);
        if (req->cmd.opcode == NVME_CMD_AGGRE_READ) {
            NvmeAggreReadCmd *acmd = (NvmeAggreReadCmd *)&req->cmd;
            uint64_t mptr = le64_to_cpu(acmd->mptr);      // Host ParamEntry
            uint64_t prp1 = le64_to_cpu(acmd->prp1);      // result buffer first page
            uint64_t prp2 = le64_to_cpu(acmd->prp2);      // result second page / PRP list
            uint32_t result_nblks = le32_to_cpu(acmd->result_nlb);
            struct ParamEntry param;
            memset(&param, 0, sizeof(param));
            if (!mptr || !prp1 || result_nblks == 0) {
                req->status = NVME_INVALID_FIELD | NVME_DNR;
                req->reqlat = 0;
                goto enqueue_done;
            }
            // femu_log("[RAW] got AGGR_READ sqid=%d mptr=0x%lx prp1=0x%lx prp2=0x%lx result_nblks=%u\n", req->sq->sqid, mptr, prp1, prp2, result_nblks);
            nvme_addr_read(n, mptr, &param, sizeof(struct ParamEntry));
            // femu_log("[RAW] param flag=%u hop=%u node_num=%u node0=%lu node1=%lu\n", param.flag, param.hop, param.node_num, param.node_id[0], param.node_id[1]);
            if (param.node_num == 0 || param.node_num > MAX_NODE_IDS) {
                femu_err("[AGGR_READ] invalid node_num=%u\n", param.node_num);
                req->status = NVME_INVALID_FIELD | NVME_DNR;
                req->reqlat = 0;
                goto enqueue_done;
            }
            uint64_t result_size = ((uint64_t)result_nblks) << FLASH_PAGE_SHIFT;
            void *result_buf = g_malloc0(result_size);
            if (!result_buf) {
                req->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
                req->reqlat = 0;
                goto enqueue_done;
            }

            uint64_t agg_lat = 0;
            bool ok = femu_execute_aggregation_param(
                n,
                &param,
                result_buf,
                result_size,
                &agg_lat
            );

            if (!ok) {
                g_free(result_buf);
                req->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
                req->reqlat = 0;
                goto enqueue_done;
            }

            uint16_t st = nvme_write_result_prp(
            n,
            req,
            prp1,
            prp2,
            result_buf,
            result_size
        );

            g_free(result_buf);

            if (st != NVME_SUCCESS) {
                req->status = st;
                req->reqlat = 0;
                goto enqueue_done;
            }
            

            if (agg_lat == 0) {
                agg_lat = NAND_READ_LATENCY;
            }

            req->status = NVME_SUCCESS;
            req->reqlat = agg_lat;
            req->expire_time = req->stime + agg_lat;
            // req->expire_time = NAND_READ_LATENCY;
        } else {
            femu_err("RAW thread got unsupported opcode=0x%x\n", req->cmd.opcode);
            req->status = NVME_INVALID_OPCODE | NVME_DNR;
            req->reqlat = 0;
            goto enqueue_done;
        }

enqueue_done:
        rc = femu_ring_enqueue(ssd->to_poller[i], (void *)&req, 1);
        if (rc != 1) {
            femu_err("RAW to_poller enqueue failed, ret=%d\n", rc);
        }
    }

    return NULL;
}

 
void ssd_init(FemuCtrl *n)
{
    struct ssd *ssd = n->ssd;
    struct ssdparams *spp = &ssd->sp;
    ftl_assert(ssd);
    ssd_init_params(spp, n);
    ssd->ch = g_malloc0(sizeof(struct ssd_channel) * spp->nchs);
    for (int i = 0; i < spp->nchs; i++) {
        ssd_init_ch(&ssd->ch[i], spp);
    }

    ssd->lftl = learned_ftl_init(spp);
    if (!ssd->lftl) {
        ftl_err("failed to init learned FTL\n");
        return;
    }

    int worker_num = n->multipoller_enabled ? n->nr_io_queues : 1;
    if (worker_num <= 0) {
        worker_num = 1;
    }

    n->nr_pollers = worker_num;
    qemu_thread_create(&ssd->lftl_thread, "FEMU-FTL-Thread", lftl_thread, n, QEMU_THREAD_JOINABLE);          
    femu_log("[] ssd_init worker_num=%d nr_pollers=%d nr_io_queues=%d\n", worker_num, n->nr_pollers, n->nr_io_queues);

    ssd->raw_threads  = g_malloc0(sizeof(QemuThread) * (worker_num + 1));    
    WorkerArg *raw_args = g_malloc0(sizeof(WorkerArg) * (worker_num + 1));

    for (int i = 1; i <= worker_num; i++) {
        char name[64];
        raw_args[i].n = n;
        raw_args[i].worker_id = i;
        snprintf(name, sizeof(name), "FEMU-RAW-%d", i);
        qemu_thread_create(&ssd->raw_threads[i],name,raw_thread, &raw_args[i], QEMU_THREAD_JOINABLE);
    }
}