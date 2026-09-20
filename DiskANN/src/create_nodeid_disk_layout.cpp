#include "create_nodeid_disk_layout.h"
#include "nodeid_nvme_writer.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cached_io.h"
#include "common_includes.h"
#include "disk_utils.h"
#include "utils.h"

namespace diskann
{

static inline uint64_t round_up_u64(uint64_t x, uint64_t align)
{
    return ((x + align - 1) / align) * align;
}

static inline uint64_t get_u64_file_size(const std::string &fname)
{
    return static_cast<uint64_t>(get_file_size(fname));
}

template <typename T>
void create_nodeid_disk_layout(const std::string &base_file, const std::string &mem_index_file, const std::string &meta_output_file, const std::string &nvme_dev_path, uint32_t nsid, const std::string &reorder_data_file)
{
    constexpr uint64_t kMagic = 0x4E4F44454944584CULL;
    constexpr uint64_t kVersion = 1;
    constexpr uint64_t kReservedNodeId = 0;
    constexpr uint64_t kDataNodeIdBase = kReservedNodeId + 1;

    constexpr uint32_t kWriteBatchSize = 4096;
    constexpr uint32_t kWriteQueueDepth = 4096;

    uint32_t npts = 0, ndims = 0;
    const size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *)&npts, sizeof(uint32_t));
    base_reader.read((char *)&ndims, sizeof(uint32_t));
    const uint64_t npts_64 = static_cast<uint64_t>(npts);
    const uint64_t ndims_64 = static_cast<uint64_t>(ndims);

    bool append_reorder_data = false;
    std::ifstream reorder_reader;
    uint32_t npts_reorder_file = 0, ndims_reorder_file = 0;

     
    #pragma region
    if (!reorder_data_file.empty())
    {
        append_reorder_data = true;
        const uint64_t reorder_file_size = get_u64_file_size(reorder_data_file);

        reorder_reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try
        {
            reorder_reader.open(reorder_data_file, std::ios::binary);
            reorder_reader.read((char *)&npts_reorder_file, sizeof(uint32_t));
            reorder_reader.read((char *)&ndims_reorder_file, sizeof(uint32_t));

            if (npts_reorder_file != npts)
            {
                throw ANNException("Mismatch in num_points between reorder data file and base file", -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            const uint64_t expected_size = 8ULL + static_cast<uint64_t>(sizeof(float)) * npts_reorder_file * ndims_reorder_file;

            if (reorder_file_size != expected_size)
            {
                throw ANNException("Discrepancy in reorder data file size", -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        catch (std::system_error &e)
        {
            throw FileException(reorder_data_file, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }
    #pragma endregion

    const uint64_t actual_file_size = get_u64_file_size(mem_index_file);
    std::cout << "Vamana index file size = " << actual_file_size << std::endl;

    std::ifstream vamana_reader(mem_index_file, std::ios::binary);

    if (!vamana_reader.is_open())
    {
        throw ANNException("Failed to open mem_index_file", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    uint64_t index_file_size = 0;
    uint32_t width_u32 = 0, medoid_u32 = 0;
    uint64_t vamana_frozen_num = 0, vamana_frozen_loc = 0;

    vamana_reader.read((char *)&index_file_size, sizeof(uint64_t));

    if (index_file_size != actual_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per meta-data. file size from file: " << index_file_size << " actual file size: " << actual_file_size;
        throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    vamana_reader.read((char *)&width_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&medoid_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&vamana_frozen_num, sizeof(uint64_t));

    if (vamana_frozen_num == 1)
    {
        vamana_frozen_loc = medoid_u32;
    }

    const uint64_t max_node_len = static_cast<uint64_t>(ndims_64 * sizeof(T)) + sizeof(uint32_t) + static_cast<uint64_t>(width_u32) * sizeof(uint32_t);
    const uint64_t node_slot_bytes = max_node_len;
    const uint64_t nodes_per_page = NodeIdNvmeWriter::kPageSize / node_slot_bytes;

    if (nodes_per_page == 0)
    {
        throw ANNException("One node is larger than one SSD page", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint32_t node_nblks = static_cast<uint32_t>((node_slot_bytes + NodeIdNvmeWriter::kPageSize - 1) / NodeIdNvmeWriter::kPageSize);
    const uint64_t node_bytes_aligned = static_cast<uint64_t>(node_nblks) * NodeIdNvmeWriter::kPageSize;

    std::cout << "npts                : " << npts_64 << std::endl;
    std::cout << "ndims               : " << ndims_64 << std::endl;
    std::cout << "medoid              : " << medoid_u32 << std::endl;
    std::cout << "width               : " << width_u32 << std::endl;
    std::cout << "max_node_len        : " << max_node_len << " B" << std::endl;
    std::cout << "node_bytes_aligned  : " << node_bytes_aligned << " B" << std::endl;
    std::cout << "node_nblks          : " << node_nblks << std::endl;
    std::cout << "nodes_per_page      : " << nodes_per_page << std::endl;
    std::cout << "data_nodeid_base    : " << kDataNodeIdBase << std::endl;
    std::cout << "write_batch_size    : " << kWriteBatchSize << std::endl;
    std::cout << "write_queue_depth   : " << kWriteQueueDepth << std::endl;

    NodeIdNvmeWriter nvme_writer(nvme_dev_path, nsid, kWriteQueueDepth);

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);

    const size_t batch_buf_bytes = static_cast<size_t>(kWriteBatchSize) * NodeIdNvmeWriter::kPageSize;
    std::unique_ptr<char[], decltype(&std::free)> batch_buf(static_cast<char *>(std::aligned_alloc(NodeIdNvmeWriter::kPageSize, batch_buf_bytes)), &std::free);

    if (!batch_buf)
    {
        throw ANNException("aligned_alloc failed for batch_buf", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::vector<NodeIdNvmeWriter::NodeReadReq> batch_reqs;
    batch_reqs.reserve(kWriteBatchSize);

    auto last_time = std::chrono::steady_clock::now();

    for (uint32_t node_id = 0; node_id < npts; node_id++)
    {
        if (node_id % 100000 == 0)
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_time).count();
            std::cout << "Packing graph node #" << node_id << ", last 100K time = " << elapsed << " s" << std::endl;
            last_time = now;
        }

        const uint32_t batch_slot = static_cast<uint32_t>(batch_reqs.size());
        char *cur_buf = batch_buf.get() + static_cast<size_t>(batch_slot) * NodeIdNvmeWriter::kPageSize;

        std::memset(cur_buf, 0, NodeIdNvmeWriter::kPageSize);

        uint32_t nnbrs = 0;
        vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

        if (!(nnbrs > 0 && nnbrs <= width_u32))
        {
            std::stringstream ss;
            ss << "Invalid nnbrs=" << nnbrs << " for node_id=" << node_id << ", width=" << width_u32;
            throw ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        uint32_t *nhood_buf = reinterpret_cast<uint32_t *>(cur_buf + ndims_64 * sizeof(T) + sizeof(uint32_t));
        vamana_reader.read((char *)nhood_buf, static_cast<size_t>(nnbrs) * sizeof(uint32_t));

        base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
        std::memcpy(cur_buf, cur_node_coords.get(), sizeof(T) * ndims_64);

        *reinterpret_cast<uint32_t *>(cur_buf + ndims_64 * sizeof(T)) = std::min(nnbrs, width_u32);

        const uint64_t ssd_nodeid = kDataNodeIdBase + node_id;

        NodeIdNvmeWriter::NodeReadReq req;
        req.node_id = ssd_nodeid;
        req.buf = cur_buf;
        req.nblks = node_nblks;
        req.data_len = max_node_len;
        req.slot_id = 0;

        batch_reqs.push_back(req);

        if (batch_reqs.size() == kWriteBatchSize)
        {
            auto status = nvme_writer.write_nodes(batch_reqs);

            if (status.size() != batch_reqs.size())
            {
                throw ANNException("NodeID batch write returned unexpected status size", -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            for (size_t i = 0; i < status.size(); i++)
            {
                if (!status[i])
                {
                    std::stringstream ss;
                    ss << "NodeID batch write failed, nodeid=" << batch_reqs[i].node_id;
                    throw ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
                }
            }

            batch_reqs.clear();
        }
    }

    if (!batch_reqs.empty())
    {
        auto status = nvme_writer.write_nodes(batch_reqs);

        if (status.size() != batch_reqs.size())
        {
            throw ANNException("Final NodeID batch write returned unexpected status size", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        for (size_t i = 0; i < status.size(); i++)
        {
            if (!status[i])
            {
                std::stringstream ss;
                ss << "Final NodeID batch write failed, nodeid=" << batch_reqs[i].node_id;
                throw ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }

        batch_reqs.clear();
    }

     
    #pragma region
    uint64_t reorder_nodeid_base = 0;
    uint64_t reorder_bytes_aligned = 0;
    uint64_t reorder_nblks = 0;

    if (append_reorder_data)
    {
        reorder_nodeid_base = static_cast<uint64_t>(kDataNodeIdBase) + npts_64;

        const uint64_t reorder_vec_bytes = static_cast<uint64_t>(ndims_reorder_file) * sizeof(float);
        reorder_bytes_aligned = round_up_u64(reorder_vec_bytes, NodeIdNvmeWriter::kPageSize);
        reorder_nblks = reorder_bytes_aligned / NodeIdNvmeWriter::kPageSize;

        std::unique_ptr<char[], decltype(&std::free)> reorder_buf(static_cast<char *>(std::aligned_alloc(NodeIdNvmeWriter::kPageSize, reorder_bytes_aligned)), &std::free);

        if (!reorder_buf)
        {
            throw ANNException("aligned_alloc failed for reorder_buf", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        for (uint32_t i = 0; i < npts_reorder_file; i++)
        {
            if (i % 100000 == 0)
            {
                std::cout << "Writing reorder node #" << i << std::endl;
            }

            std::memset(reorder_buf.get(), 0, reorder_bytes_aligned);
            reorder_reader.read(reorder_buf.get(), reorder_vec_bytes);

            const uint64_t reorder_nodeid = static_cast<uint64_t>(reorder_nodeid_base + i);
            nvme_writer.write_node(reorder_nodeid, reorder_buf.get(), static_cast<uint32_t>(reorder_nblks));
        }
    }
    #pragma endregion

    NodeDiskMeta meta{};
    meta.magic = kMagic;
    meta.version = kVersion;
    meta.npts = npts_64;
    meta.ndims = ndims_64;
    meta.medoid = medoid_u32;
    meta.width = width_u32;
    meta.max_node_len = max_node_len;
    meta.node_bytes_aligned = node_bytes_aligned;
    meta.node_nblks = node_nblks;
    meta.data_nodeid_base = kDataNodeIdBase;
    meta.append_reorder_data = append_reorder_data ? 1 : 0;
    meta.reorder_nodeid_base = reorder_nodeid_base;
    meta.node_vec_bytes = static_cast<uint64_t>(ndims_64 * sizeof(T));
    meta.node_slot_bytes = node_slot_bytes;
    meta.nodes_per_page = nodes_per_page;
    meta.reorder_ndims = ndims_reorder_file;
    meta.reorder_bytes_aligned = reorder_bytes_aligned;
    meta.reorder_nblks = reorder_nblks;
    meta.vamana_frozen_num = vamana_frozen_num;
    meta.vamana_frozen_loc = vamana_frozen_loc;

    constexpr uint64_t kMetaNodeId = 0;
    constexpr uint32_t kMetaNblks = 1;

    std::unique_ptr<char[], decltype(&std::free)> meta_buf(static_cast<char *>(std::aligned_alloc(NodeIdNvmeWriter::kPageSize, kMetaNblks * NodeIdNvmeWriter::kPageSize)), &std::free);

    if (!meta_buf)
    {
        throw ANNException("aligned_alloc failed for meta_buf", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::memset(meta_buf.get(), 0, kMetaNblks * NodeIdNvmeWriter::kPageSize);

    if (sizeof(NodeDiskMeta) > kMetaNblks * NodeIdNvmeWriter::kPageSize)
    {
        throw ANNException("NodeDiskMeta is larger than reserved meta node space", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::memcpy(meta_buf.get(), &meta, sizeof(NodeDiskMeta));
    nvme_writer.write_node(kMetaNodeId, meta_buf.get(), kMetaNblks);

    save_bin<NodeDiskMeta>(meta_output_file, &meta, 1, 1);

    std::cout << "NodeID SSD meta written to " << meta_output_file << std::endl;
}

 
template void create_nodeid_disk_layout<float>(const std::string &, const std::string &, const std::string &,
                                               const std::string &, uint32_t, const std::string &);

template void create_nodeid_disk_layout<uint8_t>(const std::string &, const std::string &, const std::string &,
                                                 const std::string &, uint32_t, const std::string &);

template void create_nodeid_disk_layout<int8_t>(const std::string &, const std::string &, const std::string &,
                                                const std::string &, uint32_t, const std::string &);

}  // namespace diskann