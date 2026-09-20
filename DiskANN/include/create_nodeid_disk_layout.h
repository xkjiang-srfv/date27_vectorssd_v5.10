#pragma once

#include <cstdint>
#include <string>

namespace diskann
{

struct NodeDiskMeta
{
    uint64_t magic;
    uint64_t version;

    uint64_t npts;
    uint64_t ndims;
    uint64_t medoid;
    uint64_t width;

    uint64_t max_node_len;
    uint64_t node_bytes_aligned;
    uint64_t node_nblks;
    uint64_t node_vec_bytes;    

      
    uint64_t node_slot_bytes;    
    uint64_t nodes_per_page;     

    uint64_t data_nodeid_base;

    uint64_t append_reorder_data;
    uint64_t reorder_nodeid_base;
    uint64_t reorder_ndims;
    uint64_t reorder_bytes_aligned;
    uint64_t reorder_nblks;

    uint64_t vamana_frozen_num;
    uint64_t vamana_frozen_loc;
};

template <typename T>
void create_nodeid_disk_layout(const std::string &base_file, const std::string &mem_index_file,
                               const std::string &meta_output_file, const std::string &nvme_dev_path, uint32_t nsid,
                               const std::string &reorder_data_file = "");

}  // namespace diskann