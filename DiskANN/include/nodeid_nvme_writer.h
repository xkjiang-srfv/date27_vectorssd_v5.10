

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <liburing.h>
#include <unordered_map>
#include <list>
#include <memory>
#include <cstdlib>


namespace diskann
{

template<typename Key, typename Value>
class LRUCache
{
private:
    using ListIt = typename std::list<std::pair<Key, Value>>::iterator;
    size_t _capacity;
    std::list<std::pair<Key, Value>> _items;
    std::unordered_map<Key, ListIt> _map;
public:
    explicit LRUCache(size_t capacity)
        : _capacity(capacity)
    {
    }
    bool exists(const Key& key)
    {
        return _map.find(key) != _map.end();
    }
    bool get(const Key& key, Value& value)
    {
        auto it = _map.find(key);
        if (it == _map.end())
        {
            return false;
        }
        _items.splice(_items.begin(), _items, it->second);
        value = it->second->second;
        return true;
    }
    void put(const Key& key, const Value& value)
    {
        auto it = _map.find(key);
        if (it != _map.end())
        {
            _items.erase(it->second);
            _map.erase(it);
        }
        _items.push_front(std::make_pair(key, value));
        _map[key] = _items.begin();
        if (_map.size() > _capacity)
        {
            auto last = _items.end();
            --last;
            _map.erase(last->first);
            _items.pop_back();
        }
    }
    void erase(const Key& key)
    {
        auto it = _map.find(key);
        if (it == _map.end())
        {
            return;
        }
        _items.erase(it->second);
        _map.erase(it);
    }
    void clear()
    {
        _items.clear();
        _map.clear();
    }
};

class NodeIdNvmeWriter       
{
   public:
    static constexpr uint32_t kPageSize = 4096;                  
    static constexpr uint8_t NVME_CMD_NODEID_WRITE = 0x81;       
    static constexpr uint8_t NVME_CMD_NODEID_READ = 0x82;        
    static constexpr uint64_t AGG_SPECIAL_NODEID = 0xFFFFFFFFFFFFFFFFULL;
    static constexpr uint32_t AGG_MAX_NODE_IDS = 16;
    static constexpr uint32_t AGG_CMD_MAGIC = 0x41474743;
    static constexpr uint32_t AGG_RES_MAGIC = 0x41474752;
    static constexpr uint64_t AGG_RESULT_CACHE_BYTES_PER_SLOT = 128ULL * 1024;
    static constexpr uint32_t AGG_MAX_SLOTS = 64;
    static constexpr uint64_t AGG_RESULT_CACHE_BYTES = AGG_RESULT_CACHE_BYTES_PER_SLOT * AGG_MAX_SLOTS;
    static constexpr uint32_t DISKANN_MAX_DEGREE_FOR_AGG = 64;
    static constexpr uint32_t DISKANN_NODE_DIM = 100;
    static constexpr uint8_t NVME_CMD_AGGRE_READ = 0x83;


    
    struct NodeReadReq                                           
    {   
        uint64_t node_id;                                        
        void *buf;                                               
        uint32_t nblks;                                          
        uint32_t data_len;               
        uint32_t slot_id;
    };

    struct AggCmd {
        uint32_t flag;      // 0 = DFS, 1 = batch
        uint32_t hop;       // batch mode unused
        uint32_t node_num;
        uint64_t node_id[AGG_MAX_NODE_IDS];
    };

    struct AggResultMeta {
        uint32_t magic;
        uint32_t status;
        uint64_t result_size;
        uint64_t total_latency;
        uint64_t read_latency;
        uint64_t write_latency;
        uint32_t read_cnt;
        uint32_t write_cnt;
    };

    struct AggPrefetchNode
    {
        uint64_t node_id;
        uint32_t degree;
        int8_t vector[DISKANN_NODE_DIM];   // SIFT100M: uint8, 128 dims
        uint32_t neighbors[DISKANN_MAX_DEGREE_FOR_AGG];
    };

    struct AggCachedNode {
        std::vector<int8_t> vec;
        uint32_t degree;
        std::vector<uint32_t> neighbors;
    };

    struct PendingAggReq {
        bool valid = false;
        uint64_t user_data = 0;
        uint32_t result_nblks = 0;
        size_t result_bytes = 0;
        std::unique_ptr<char[], decltype(&std::free)> param_buf{nullptr, &std::free};
        std::unique_ptr<char[], decltype(&std::free)> result_buf{nullptr, &std::free};
    };

    PendingAggReq _pending_agg;
    
    bool submit_aggregation_prefetch_batch_async(
        const std::vector<uint64_t> &physical_node_ids);

    bool wait_aggregation_prefetch_batch(
        std::vector<NodeIdNvmeWriter::AggPrefetchNode> &out_nodes);



public:
    NodeIdNvmeWriter(const std::string& dev_path, uint32_t nsid,uint32_t queue_depth = 1024);
    ~NodeIdNvmeWriter();
    NodeIdNvmeWriter(const NodeIdNvmeWriter&) = delete;
    NodeIdNvmeWriter& operator=( const NodeIdNvmeWriter&) = delete;

public:
    void write_node(uint64_t node_id, const void* buf, uint32_t nblks,uint32_t slot_id = 0);
    void read_node(uint64_t node_id, void* buf, uint32_t nblks,uint32_t slot_id = 0);
    std::vector<bool> read_nodes(const std::vector<NodeReadReq>& reqs);
    std::vector<bool> write_nodes(const std::vector<NodeReadReq>& reqs);
    bool aggregation_prefetch_batch(const std::vector<uint64_t>& physical_node_ids, std::vector<AggPrefetchNode>& out_nodes);



public:
    bool get_cached_agg_node(uint32_t logical_id, AggCachedNode& out)
    {
        return _agg_cache.get(logical_id, out);
    }

    bool has_cached_agg_node(uint32_t logical_id)
    {
        return _agg_cache.exists(logical_id);
    }

    void put_cached_agg_node(uint32_t logical_id, const AggCachedNode& node)
    {
        _agg_cache.put(logical_id, node);
    }

    void erase_cached_agg_node(uint32_t logical_id)
    {
        _agg_cache.erase(logical_id);

    }

    void clear_agg_cache()
    {
        _agg_cache.clear();
    }


private:
    int _fd;                                 
    uint32_t _nsid;                          
    io_uring _ring;                          
    bool _ring_inited;                       
    uint32_t _queue_depth;                   

private:
    LRUCache<uint32_t, AggCachedNode> _agg_cache{32};

private:
    void submit_cmd(uint8_t opcode, uint64_t node_id, void *buf, uint32_t nblks,uint32_t slot_id = 0);        
    uint64_t _batch_seq = 0;
    std::vector<bool> submit_cmd_batch(uint8_t opcode, const std::vector<NodeReadReq> &reqs);    
};

}  // namespace diskann




