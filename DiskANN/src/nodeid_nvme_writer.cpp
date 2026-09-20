#include "nodeid_nvme_writer.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/nvme_ioctl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sstream>
#include <iostream>

#include <memory>
#include <cstdlib>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <atomic>
#include <chrono>

static inline uint64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}


namespace diskann
{

static inline std::runtime_error make_runtime_error(const std::string &msg)  
{
    return std::runtime_error(msg + ": " + std::strerror(errno));
}

NodeIdNvmeWriter::NodeIdNvmeWriter(const std::string &dev_path, uint32_t nsid, uint32_t queue_depth)
    : _fd(-1), _nsid(nsid), _ring_inited(false), _queue_depth(queue_depth)           
{
    _fd = ::open(dev_path.c_str(), O_RDWR);                      
    if (_fd < 0)
    {
        throw make_runtime_error("open nvme device failed");
    }

    io_uring_params p;                                          
    std::memset(&p, 0, sizeof(p));
    p.flags |= IORING_SETUP_SQE128;                              
    p.flags |= IORING_SETUP_CQE32;                               

    int ret = io_uring_queue_init_params(queue_depth, &_ring, &p);   
    if (ret < 0)
    {
        ::close(_fd);
        _fd = -1;
        errno = -ret;
        throw make_runtime_error("io_uring_queue_init_params failed");
    }
    _ring_inited = true;
}

NodeIdNvmeWriter::~NodeIdNvmeWriter()                            
{
    if (_ring_inited)
    {
        io_uring_queue_exit(&_ring);
        _ring_inited = false;
    }
    if (_fd >= 0)
    {
        ::close(_fd);
        _fd = -1;
    }
}

namespace
{
inline void validate_nblks(uint32_t nblks)
{
    if (nblks == 0)
    {
        throw std::runtime_error("nblks must be > 0");
    }
}

inline uint32_t calc_data_len(uint32_t nblks, uint32_t page_size)
{
    const uint64_t data_len_64 = static_cast<uint64_t>(nblks) * page_size;
    if (data_len_64 > UINT32_MAX)
    {
        throw std::runtime_error("data_len too large");
    }
    return static_cast<uint32_t>(data_len_64);
}
}  // namespace


bool NodeIdNvmeWriter::aggregation_prefetch_batch(
    const std::vector<uint64_t>& physical_node_ids,
    std::vector<NodeIdNvmeWriter::AggPrefetchNode>& out_nodes)
{
    out_nodes.clear();

    if (physical_node_ids.empty()) {
        return true;
    }

    if (physical_node_ids.size() < 8) {
        return true;
    }

    if (physical_node_ids.size() > AGG_MAX_NODE_IDS) {
        throw std::runtime_error("aggregation_prefetch_batch: too many node ids");
    }

    const uint32_t result_nblks =
        static_cast<uint32_t>(
            (sizeof(AggResultMeta) +
             physical_node_ids.size() * sizeof(AggPrefetchNode) +
             kPageSize - 1) /
            kPageSize);

    const size_t result_bytes =
        static_cast<size_t>(result_nblks) * kPageSize;

    std::unique_ptr<char[], decltype(&std::free)> param_buf(
        static_cast<char *>(std::aligned_alloc(kPageSize, kPageSize)),
        &std::free);

    std::unique_ptr<char[], decltype(&std::free)> result_buf(
        static_cast<char *>(std::aligned_alloc(kPageSize, result_bytes)),
        &std::free);

    if (!param_buf || !result_buf) {
        throw std::runtime_error("aggregation_prefetch_batch: aligned_alloc failed");
    }

    std::memset(param_buf.get(), 0, kPageSize);
    std::memset(result_buf.get(), 0, result_bytes);

    AggCmd cmd{};
    cmd.flag = 1;
    cmd.hop = 0;
    cmd.node_num = static_cast<uint32_t>(physical_node_ids.size());

    for (size_t i = 0; i < physical_node_ids.size(); i++) {
        cmd.node_id[i] = physical_node_ids[i];
    }

    std::memcpy(param_buf.get(), &cmd, sizeof(cmd));

    io_uring_sqe *sqe = io_uring_get_sqe(&_ring);
    if (!sqe) {
        throw std::runtime_error("aggregation_prefetch_batch: io_uring_get_sqe failed");
    }

    io_uring_prep_uring_cmd(sqe, NVME_URING_CMD_IO, _fd);

    struct nvme_uring_cmd nvme_cmd;
    std::memset(&nvme_cmd, 0, sizeof(nvme_cmd));

    nvme_cmd.opcode = NVME_CMD_AGGRE_READ;
    nvme_cmd.nsid = _nsid;

    nvme_cmd.metadata = reinterpret_cast<uint64_t>(param_buf.get());
    nvme_cmd.metadata_len = sizeof(AggCmd);

    nvme_cmd.addr = reinterpret_cast<uint64_t>(result_buf.get());
    nvme_cmd.data_len = static_cast<uint32_t>(result_bytes);

    nvme_cmd.cdw10 = result_nblks;

    std::memcpy(sqe->cmd, &nvme_cmd, sizeof(nvme_cmd));

    const uint64_t AGG_MAGIC = 0xA83ULL;
    const uint64_t agg_seq = (++_batch_seq) & 0xffffffffULL;
    const uint64_t my_user_data = (AGG_MAGIC << 48) | agg_seq;

    sqe->user_data = my_user_data;

    int ret = io_uring_submit(&_ring);
    if (ret < 0) {
        errno = -ret;
        throw make_runtime_error("aggregation_prefetch_batch: io_uring_submit failed");
    }

    if (ret != 1) {
        throw std::runtime_error("aggregation_prefetch_batch: short submit");
    }

    int my_cqe_res = 0;
    bool got_my_cqe = false;

    while (!got_my_cqe) {
        io_uring_cqe *cqe = nullptr;

        ret = io_uring_wait_cqe(&_ring, &cqe);
        if (ret < 0) {
            errno = -ret;
            throw make_runtime_error("aggregation_prefetch_batch: io_uring_wait_cqe failed");
        }

        const uint64_t got_user_data = cqe->user_data;
        const int cqe_res = cqe->res;

        io_uring_cqe_seen(&_ring, cqe);

        if (got_user_data != my_user_data) {
            continue;
        }

        my_cqe_res = cqe_res;
        got_my_cqe = true;
    }

    if (my_cqe_res < 0) {
        errno = -my_cqe_res;
        throw make_runtime_error("aggregation_prefetch_batch: AGGRE_READ failed");
    }

    auto *meta = reinterpret_cast<AggResultMeta *>(result_buf.get());

    if (meta->magic != AGG_RES_MAGIC) {
        std::stringstream ss;
        ss << "aggregation_prefetch_batch: invalid magic, got 0x"
           << std::hex << meta->magic
           << ", expected 0x" << AGG_RES_MAGIC;
        throw std::runtime_error(ss.str());
    }

    if (meta->status != 0 || meta->result_size == 0) {
        return false;
    }

    const uint64_t max_payload = result_bytes - sizeof(AggResultMeta);

    if (meta->result_size > max_payload) {
        throw std::runtime_error("aggregation_prefetch_batch: result too large");
    }

    if (meta->result_size % sizeof(AggPrefetchNode) != 0) {
        throw std::runtime_error("aggregation_prefetch_batch: unaligned result size");
    }

    const uint64_t node_cnt = meta->result_size / sizeof(AggPrefetchNode);

    auto *nodes =
        reinterpret_cast<AggPrefetchNode *>(
            result_buf.get() + sizeof(AggResultMeta));

    out_nodes.assign(nodes, nodes + node_cnt);

    return true;
}

bool NodeIdNvmeWriter::submit_aggregation_prefetch_batch_async(
    const std::vector<uint64_t> &physical_node_ids)
{
    if (physical_node_ids.empty()) {
        return false;
    }

    if (physical_node_ids.size() < 8) {
        return false;
    }

    if (physical_node_ids.size() > AGG_MAX_NODE_IDS) {
        throw std::runtime_error("submit_aggregation_prefetch_batch_async: too many node ids");
    }

    if (_pending_agg.valid) {
        throw std::runtime_error("submit_aggregation_prefetch_batch_async: previous agg still pending");
    }

    const uint32_t result_nblks =
        static_cast<uint32_t>(
            (sizeof(AggResultMeta) +
             physical_node_ids.size() * sizeof(AggPrefetchNode) +
             kPageSize - 1) /
            kPageSize);

    const size_t result_bytes =
        static_cast<size_t>(result_nblks) * kPageSize;

    std::unique_ptr<char[], decltype(&std::free)> param_buf(
        static_cast<char *>(std::aligned_alloc(kPageSize, kPageSize)),
        &std::free);

    std::unique_ptr<char[], decltype(&std::free)> result_buf(
        static_cast<char *>(std::aligned_alloc(kPageSize, result_bytes)),
        &std::free);

    if (!param_buf || !result_buf) {
        throw std::runtime_error("submit_aggregation_prefetch_batch_async: aligned_alloc failed");
    }

    std::memset(param_buf.get(), 0, kPageSize);
    std::memset(result_buf.get(), 0, result_bytes);

    AggCmd cmd{};
    cmd.flag = 1;
    cmd.hop = 0;
    cmd.node_num = static_cast<uint32_t>(physical_node_ids.size());

    for (size_t i = 0; i < physical_node_ids.size(); i++) {
        cmd.node_id[i] = physical_node_ids[i];
    }

    std::memcpy(param_buf.get(), &cmd, sizeof(cmd));

    io_uring_sqe *sqe = io_uring_get_sqe(&_ring);
    if (!sqe) {
        throw std::runtime_error("submit_aggregation_prefetch_batch_async: io_uring_get_sqe failed");
    }

    io_uring_prep_uring_cmd(sqe, NVME_URING_CMD_IO, _fd);

    struct nvme_uring_cmd nvme_cmd;
    std::memset(&nvme_cmd, 0, sizeof(nvme_cmd));

    nvme_cmd.opcode = NVME_CMD_AGGRE_READ;
    nvme_cmd.nsid = _nsid;

    nvme_cmd.metadata = reinterpret_cast<uint64_t>(param_buf.get());
    nvme_cmd.metadata_len = sizeof(AggCmd);

    nvme_cmd.addr = reinterpret_cast<uint64_t>(result_buf.get());
    nvme_cmd.data_len = static_cast<uint32_t>(result_bytes);

    nvme_cmd.cdw10 = result_nblks;

    std::memcpy(sqe->cmd, &nvme_cmd, sizeof(nvme_cmd));

    const uint64_t AGG_MAGIC = 0xA83ULL;
    const uint64_t agg_seq = (++_batch_seq) & 0xffffffffULL;
    const uint64_t my_user_data = (AGG_MAGIC << 48) | agg_seq;

    sqe->user_data = my_user_data;

    int ret = io_uring_submit(&_ring);
    if (ret < 0) {
        errno = -ret;
        throw make_runtime_error("submit_aggregation_prefetch_batch_async: io_uring_submit failed");
    }

    if (ret != 1) {
        throw std::runtime_error("submit_aggregation_prefetch_batch_async: short submit");
    }

    _pending_agg.valid = true;
    _pending_agg.user_data = my_user_data;
    _pending_agg.result_nblks = result_nblks;
    _pending_agg.result_bytes = result_bytes;
    _pending_agg.param_buf = std::move(param_buf);
    _pending_agg.result_buf = std::move(result_buf);

    return true;
}

bool NodeIdNvmeWriter::wait_aggregation_prefetch_batch(
    std::vector<NodeIdNvmeWriter::AggPrefetchNode> &out_nodes)
{
    out_nodes.clear();

    if (!_pending_agg.valid) {
        return false;
    }

    int my_cqe_res = 0;
    bool got_my_cqe = false;

    while (!got_my_cqe) {
        io_uring_cqe *cqe = nullptr;

        int ret = io_uring_wait_cqe(&_ring, &cqe);
        if (ret < 0) {
            errno = -ret;
            throw make_runtime_error("wait_aggregation_prefetch_batch: io_uring_wait_cqe failed");
        }

        const uint64_t got_user_data = cqe->user_data;
        const int cqe_res = cqe->res;

        io_uring_cqe_seen(&_ring, cqe);

        if (got_user_data != _pending_agg.user_data) {
            continue;
        }

        my_cqe_res = cqe_res;
        got_my_cqe = true;
    }

    if (my_cqe_res < 0) {
        errno = -my_cqe_res;
        _pending_agg.valid = false;
        throw make_runtime_error("wait_aggregation_prefetch_batch: AGGRE_READ failed");
    }

    auto *meta =
        reinterpret_cast<AggResultMeta *>(_pending_agg.result_buf.get());

    if (meta->magic != AGG_RES_MAGIC) {
        _pending_agg.valid = false;
        throw std::runtime_error("wait_aggregation_prefetch_batch: invalid magic");
    }

    if (meta->status != 0 || meta->result_size == 0) {
        _pending_agg.valid = false;
        return false;
    }

    const uint64_t max_payload =
        _pending_agg.result_bytes - sizeof(AggResultMeta);

    if (meta->result_size > max_payload) {
        _pending_agg.valid = false;
        throw std::runtime_error("wait_aggregation_prefetch_batch: result too large");
    }

    if (meta->result_size % sizeof(AggPrefetchNode) != 0) {
        _pending_agg.valid = false;
        throw std::runtime_error("wait_aggregation_prefetch_batch: unaligned result size");
    }

    const uint64_t node_cnt =
        meta->result_size / sizeof(AggPrefetchNode);

    auto *nodes =
        reinterpret_cast<AggPrefetchNode *>(
            _pending_agg.result_buf.get() + sizeof(AggResultMeta));

    out_nodes.assign(nodes, nodes + node_cnt);

    _pending_agg.valid = false;
    _pending_agg.user_data = 0;
    _pending_agg.result_nblks = 0;
    _pending_agg.result_bytes = 0;
    _pending_agg.param_buf.reset();
    _pending_agg.result_buf.reset();

    return true;
}

void NodeIdNvmeWriter::submit_cmd(uint8_t opcode, uint64_t node_id, void *buf, uint32_t nblks,uint32_t slot_id)
{
    NodeReadReq req;
    req.node_id = node_id;
    req.buf = buf;
    req.nblks = nblks;
    req.slot_id = slot_id;
    req.data_len = 0;

    auto status = submit_cmd_batch(opcode, std::vector<NodeReadReq>{req});
    if (status.empty() || !status[0])
    {
        throw std::runtime_error("submit_cmd failed");
    }
}

std::vector<bool> NodeIdNvmeWriter::submit_cmd_batch(uint8_t opcode, const std::vector<NodeReadReq> &reqs)
{
    std::vector<bool> status(reqs.size(), false);

    if (reqs.empty()) {
        return status;
    }

    if (_queue_depth == 0) {
        throw std::runtime_error("queue_depth must be > 0");
    }

    const uint64_t BATCH_MAGIC = 0xB17ULL;
    const uint64_t batch_id = (++_batch_seq) & 0xffffULL;

    auto make_user_data = [&](uint32_t idx) -> uint64_t {
        return (BATCH_MAGIC << 48) | (batch_id << 32) | static_cast<uint64_t>(idx);
    };

    auto is_my_cqe = [&](uint64_t user_data, size_t &idx_out) -> bool {
        uint64_t magic = user_data >> 48;
        uint64_t bid = (user_data >> 32) & 0xffffULL;
        uint64_t idx = user_data & 0xffffffffULL;

        if (magic != BATCH_MAGIC || bid != batch_id) {
            return false;
        }

        if (idx >= reqs.size()) {
            return false;
        }

        idx_out = static_cast<size_t>(idx);
        return true;
    };

    size_t submitted = 0;
    size_t completed = 0;
    size_t inflight = 0;

    auto prepare_one = [&](size_t req_idx) {
        const auto &r = reqs[req_idx];
        validate_nblks(r.nblks);

        const uint32_t max_data_len = calc_data_len(r.nblks, kPageSize);
        const uint32_t data_len = (r.data_len != 0) ? r.data_len : max_data_len;

        if (data_len == 0 || data_len > max_data_len) {
            throw std::runtime_error("submit_cmd_batch: invalid data_len");
        }

        if (data_len == 0 || data_len > max_data_len) {
            throw std::runtime_error("submit_cmd_batch: invalid data_len");
        }

        io_uring_sqe *sqe = io_uring_get_sqe(&_ring);
        if (!sqe) {
            throw std::runtime_error("submit_cmd_batch: io_uring_get_sqe failed");
        }

        io_uring_prep_uring_cmd(sqe, NVME_URING_CMD_IO, _fd);

        struct nvme_uring_cmd cmd;
        std::memset(&cmd, 0, sizeof(cmd));

        cmd.opcode = opcode;
        cmd.nsid = _nsid;
        cmd.addr = reinterpret_cast<uint64_t>(r.buf);
        cmd.data_len = data_len;
        cmd.cdw10 = static_cast<uint32_t>(r.node_id & 0xffffffffULL);
        cmd.cdw11 = static_cast<uint32_t>(r.node_id >> 32);
        cmd.cdw12 = r.nblks;
        cmd.cdw13 = r.slot_id;

        std::memcpy(sqe->cmd, &cmd, sizeof(cmd));
        sqe->user_data = make_user_data(static_cast<uint32_t>(req_idx));
    };

    auto submit_n = [&](size_t want_submit) {
        if (want_submit == 0) {
            return;
        }
        // fprintf(stderr, "[NODEID_SUBMIT_BEFORE] ts=%llu want=%zu\n", (unsigned long long)now_ns(), want_submit);
        int ret = io_uring_submit(&_ring);
        // fprintf(stderr, "[NODEID_SUBMIT_AFTER] ts=%llu ret=%d\n", (unsigned long long)now_ns(), ret);
        if (ret < 0) {
            errno = -ret;
            throw make_runtime_error("submit_cmd_batch: io_uring_submit failed");
        }

        if (static_cast<size_t>(ret) != want_submit) {
            throw std::runtime_error(
                "submit_cmd_batch: short submit, expected " +
                std::to_string(want_submit) +
                " but got " +
                std::to_string(ret));
        }
    };

    auto fill_and_submit = [&]() {
        size_t prepared_now = 0;

        while (submitted + prepared_now < reqs.size() && inflight + prepared_now < _queue_depth) {
            prepare_one(submitted + prepared_now);
            prepared_now++;
        }

        submit_n(prepared_now);
        submitted += prepared_now;
        inflight += prepared_now;
    };

    fill_and_submit();

    while (completed < reqs.size()) {
        io_uring_cqe *cqe = nullptr;

        int ret = io_uring_wait_cqe(&_ring, &cqe);
        if (ret < 0) {
            errno = -ret;
            throw make_runtime_error("submit_cmd_batch: io_uring_wait_cqe failed");
        }

        size_t completed_now = 0;

        auto handle_cqe = [&](io_uring_cqe *cur) {
            size_t idx = 0;
            uint64_t user_data = cur->user_data;
            int res = cur->res;

            io_uring_cqe_seen(&_ring, cur);

            if (!is_my_cqe(user_data, idx)) {
                return;
            }

            status[idx] = (res >= 0);
            completed_now++;
        };

        handle_cqe(cqe);

        constexpr unsigned MAX_EXTRA_CQE = 63;
        for (unsigned i = 0; i < MAX_EXTRA_CQE; i++) {
            io_uring_cqe *extra = nullptr;
            ret = io_uring_peek_cqe(&_ring, &extra);

            if (ret == -EAGAIN || extra == nullptr) {
                break;
            }

            if (ret < 0) {
                errno = -ret;
                throw make_runtime_error("submit_cmd_batch: io_uring_peek_cqe failed");
            }

            handle_cqe(extra);
        }

        if (completed_now > inflight) {
            throw std::runtime_error("submit_cmd_batch: completed_now > inflight");
        }

        completed += completed_now;
        inflight -= completed_now;

        fill_and_submit();
    }

    return status;
}

void NodeIdNvmeWriter::write_node(uint64_t node_id, const void *buf, uint32_t nblks,uint32_t slot_id)
{
    submit_cmd(NVME_CMD_NODEID_WRITE, node_id, const_cast<void *>(buf), nblks, slot_id);
}

void NodeIdNvmeWriter::read_node(uint64_t node_id, void *buf, uint32_t nblks,uint32_t slot_id)
{
    submit_cmd(NVME_CMD_NODEID_READ, node_id, buf, nblks, slot_id);
}

std::vector<bool> NodeIdNvmeWriter::read_nodes(const std::vector<NodeReadReq> &reqs)
{
    return submit_cmd_batch(NVME_CMD_NODEID_READ, reqs);
}

std::vector<bool> NodeIdNvmeWriter::write_nodes(const std::vector<NodeReadReq> &reqs)
{
    return submit_cmd_batch(NVME_CMD_NODEID_WRITE, reqs);
}

}  // namespace diskann