// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma region
#include "common_includes.h"

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"
#include "nodeid_nvme_writer.h"
#include "create_nodeid_disk_layout.h"
#define USE_NODEID_LAYOUT 1

#include <fcntl.h>
#include <unistd.h>

static int trace_fd=-1;

static inline void trace_mark(const char* msg)
{
    if(trace_fd<0)
        trace_fd=open(
            "/sys/kernel/debug/tracing/trace_marker",
            O_WRONLY);

    if(trace_fd>=0)
        write(trace_fd,msg,strlen(msg));
}

static inline uint64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}


#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, diskann::Metric m)
    : reader(fileReader), metric(m), _thread_data(nullptr)
{
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT>
inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    const uint64_t vec_bytes = _use_nodeid_layout ? _node_vec_bytes : _disk_bytes_per_point;
    return reinterpret_cast<uint32_t *>(node_buf + vec_bytes);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            if (this->_use_nodeid_layout)
            {
                data->nodeid_reader.reset(
                    new diskann::NodeIdNvmeWriter(this->_nvme_dev_path, this->_nvme_nsid));

                data->agg_nodeid_reader.reset(
                    new diskann::NodeIdNvmeWriter(this->_nvme_dev_path, this->_nvme_nsid));
            }
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(
    NodeIdNvmeWriter* nodeid_reader,
    const std::vector<uint32_t> &node_ids,
    std::vector<T *> &coord_buffers,
    std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    if (_use_nodeid_layout)
    {
        return read_nodes_nodeid(nodeid_reader, node_ids, coord_buffers, nbr_buffers);
    }
    return read_nodes_file(node_ids, coord_buffers, nbr_buffers);
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes_nodeid(
    NodeIdNvmeWriter* nodeid_reader,
    const std::vector<uint32_t> &node_ids,
    std::vector<T *> &coord_buffers,
    std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<bool> retval(node_ids.size(), false);
    if (node_ids.empty())
    {
        return retval;
    }

    if (!nodeid_reader)
    {
        throw ANNException("Thread-local NodeID reader is not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_node_nblks == 0)
    {
        throw ANNException("Invalid _node_nblks=0 for NodeID layout", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_node_vec_bytes == 0)
    {
        throw ANNException("Invalid _node_vec_bytes=0 for NodeID layout", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const size_t per_node_bytes =
        static_cast<size_t>(_node_nblks) * NodeIdNvmeWriter::kPageSize;

    std::unique_ptr<char[], decltype(&std::free)> bigbuf(
        static_cast<char *>(std::aligned_alloc(NodeIdNvmeWriter::kPageSize,
                                               node_ids.size() * per_node_bytes)),
        &std::free);

    if (!bigbuf)
    {
        throw ANNException("aligned_alloc failed in read_nodes_nodeid", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::memset(bigbuf.get(), 0, node_ids.size() * per_node_bytes);

    std::vector<NodeIdNvmeWriter::NodeReadReq> reqs;
    reqs.reserve(node_ids.size());

    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        NodeIdNvmeWriter::NodeReadReq req;
        uint64_t physical_nodeid = static_cast<uint64_t>(_data_nodeid_base) + static_cast<uint64_t>(node_ids[i]);
        req.node_id = physical_nodeid;
        req.buf = bigbuf.get() + i * per_node_bytes;
        req.nblks = static_cast<uint32_t>(_node_nblks);
        req.data_len = static_cast<uint32_t>(_max_node_len);
        req.slot_id = 0;
        reqs.push_back(req);
    }

    auto status = nodeid_reader->read_nodes(reqs);

    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        retval[i] = status[i];
        if (!status[i])
        {
            continue;
        }

        char *node_buf = bigbuf.get() + i * per_node_bytes;

        if (coord_buffers[i] != nullptr)
        {
            std::memcpy(coord_buffers[i], node_buf, _node_vec_bytes);
        }

        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood =
                reinterpret_cast<uint32_t *>(node_buf + _node_vec_bytes);

            uint32_t num_nbrs = *node_nhood;
            if (num_nbrs > _max_degree)
            {
                std::stringstream ss;
                ss << "Corrupted NodeID node: num_nbrs=" << num_nbrs
                   << " > max_degree=" << _max_degree
                   << ", logical node_id=" << node_ids[i];
                throw ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            nbr_buffers[i].first = num_nbrs;
            std::memcpy(nbr_buffers[i].second, node_nhood + 1,
                        num_nbrs * sizeof(uint32_t));
        }
    }

    return retval;
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes_file(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    reader->read(read_reqs, ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif

        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

        if (coord_buffers[i] != nullptr)
        {
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
        }

        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

#pragma endregion



#pragma region
#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    if (num_nodes_to_cache >= this->_num_points)
    {
        // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
        node_list.resize(this->_num_points);
        for (uint32_t i = 0; i < this->_num_points; ++i)
        {
            node_list[i] = i;
        }
        return;
    }

    this->_count_visited_nodes = true;
    this->_node_visit_counter.clear();
    this->_node_visit_counter.resize(this->_num_points);
    for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    {
        this->_node_visit_counter[i].first = i;
        this->_node_visit_counter[i].second = 0;
    }

    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

    bool filtered_search = false;
    std::vector<LabelT> random_query_filters(sample_num);
    if (_filter_to_medoid_ids.size() != 0)
    {
        filtered_search = true;
        generate_random_labels(random_query_filters, (uint32_t)sample_num, nthreads);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        auto &label_for_search = random_query_filters[i];
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                           tmp_result_dists.data() + i, beamwidth, filtered_search, label_for_search, false);
    }

    std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    num_nodes_to_cache = std::min(num_nodes_to_cache, this->_node_visit_counter.size());
    node_list.reserve(num_nodes_to_cache);
    for (uint64_t i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->_node_visit_counter[i].first);
    }
    this->_count_visited_nodes = false;

    diskann::aligned_free(samples);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    tsl::robin_set<uint32_t> node_set;

    // Do not cache more than 10% of the nodes in the index
    uint64_t tenp_nodes = (uint64_t)(std::round(this->_num_points * 0.1));
    if (num_nodes_to_cache > tenp_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                      << "(10 percent of total nodes:" << this->_num_points << ")" << std::endl;
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    diskann::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;

    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++)
    {
        cur_level->insert(_medoids[miter]);
    }

    if ((_filter_to_medoid_ids.size() > 0) && (cur_level->size() < num_nodes_to_cache))
    {
        for (auto &x : _filter_to_medoid_ids)
        {
            for (auto &y : x.second)
            {
                cur_level->insert(y);
                if (cur_level->size() == num_nodes_to_cache)
                    break;
            }
            if (cur_level->size() == num_nodes_to_cache)
                break;
        }
    }

    uint64_t lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<uint32_t> nodes_to_expand;

        for (const uint32_t &id : *prev_level)
        {
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle)
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        else
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        diskann::cout << "Level: " << lvl << std::flush;
        bool finish_flag = false;

        uint64_t block_size = 1024;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), block_size);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            size_t start = block * block_size;
            size_t end = (std::min)((block + 1) * block_size, nodes_to_expand.size());

            std::vector<uint32_t> nodes_to_read;
            std::vector<T *> coord_buffers(end - start, nullptr);
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
            }

            // issue read requests
            auto read_status = read_nodes(_nodeid_reader.get(), nodes_to_read, coord_buffers, nbr_buffers);

            // process each nhood buf
            for (uint32_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == false)
                {
                    continue;
                }
                else
                {
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;

                    // explore next level
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        if (node_set.find(nbrs[j]) == node_set.end())
                        {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                        {
                            finish_flag = true;
                        }
                    }
                }
                delete[] nbr_buffers[i].second;
            }
        }

        diskann::cout << ". #nodes: " << node_set.size() - prev_node_set_size
                      << ", #nodes thus far: " << node_set.size() << std::endl;
        prev_node_set_size = node_set.size();
        lvl++;
    }

    assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: " << node_list.size()
                  << std::endl;
    diskann::cout << "done" << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(_nodeid_reader.get(), nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < _data_dim; i++)
                    _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                _disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (_centroid_data + cur_m * _aligned_dim));
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix)
{
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";                     
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";        
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";                    
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#else
    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#endif
}
#pragma endregion


#pragma region
#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath)
{
#endif
     










    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

     


    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

     

 
    this->_disk_index_file = _disk_index_file;
    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;

     


#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint8_t>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;

     


#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file))
    {
        FileContent &content_labels = files.getContent(labels_file);
        std::stringstream infile(std::string((const char *)content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file))
    {
        std::ifstream infile(labels_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + labels_file, -1);
        }
#endif
        parse_label_file(infile, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->_num_points);

#ifndef EXEC_ENV_OLS
        infile.close();
#endif

#ifdef EXEC_ENV_OLS
        FileContent &content_labels_map = files.getContent(labels_map_file);
        std::stringstream map_reader(std::string((const char *)content_labels_map._content, content_labels_map._size));
#else
        std::ifstream map_reader(labels_map_file);
#endif
        _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
        map_reader.close();
#endif

#ifdef EXEC_ENV_OLS
        if (files.fileExists(labels_to_medoids))
        {
            FileContent &content_labels_to_meoids = files.getContent(labels_to_medoids);
            std::stringstream medoid_stream(
                std::string((const char *)content_labels_to_meoids._content, content_labels_to_meoids._size));
#else
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
#endif
            std::string line, token;

            _filter_to_medoid_ids.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    uint32_t cnt = 0;
                    std::vector<uint32_t> medoids;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = (LabelT)std::stoul(token);
                        else
                            medoids.push_back((uint32_t)stoul(token));
                        cnt++;
                    }
                    _filter_to_medoid_ids[label].swap(medoids);
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
        if (files.fileExists(univ_label_file))
        {
            FileContent &content_univ_label = files.getContent(univ_label_file);
            std::stringstream universal_label_reader(
                std::string((const char *)content_univ_label._content, content_univ_label._size));
#else
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
#endif
            std::string univ_label;
            universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
            universal_label_reader.close();
#endif
            LabelT label_as_num = (LabelT)std::stoul(univ_label);
            set_universal_label(label_as_num);
        }

#ifdef EXEC_ENV_OLS
        if (files.fileExists(dummy_map_file))
        {
            FileContent &content_dummy_map = files.getContent(dummy_map_file);
            std::stringstream dummy_map_stream(
                std::string((const char *)content_dummy_map._content, content_dummy_map._size));
#else
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
#endif
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t dummy_id;
                uint32_t real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (uint32_t)stoul(token);
                    else
                        real_id = (uint32_t)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<uint32_t>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
#ifndef EXEC_ENV_OLS
            dummy_map_stream.close();
#endif
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

     


#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

     


    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

     


    std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
#ifdef EXEC_ENV_OLS
    if (files.fileExists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
    if (file_exists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
        _disk_bytes_per_point =
            _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << _disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }
    size_t medoid_id_on_file = 0;
    #ifdef EXEC_ENV_OLS
        char *bytes = nullptr;
    #else
        std::ifstream index_metadata;
    #endif
     


    // read index metadata
    if (_use_nodeid_layout)
        {
            if (_nvme_dev_path.empty())
            {
                throw diskann::ANNException("NodeID layout enabled but nvme_dev_path is empty", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
            }

            constexpr uint64_t kMetaNodeId = 0;
            constexpr uint32_t kMetaNblks = 1;
            constexpr uint64_t kExpectedMagic = 0x4E4F44454944584CULL;  // "NODEIDXL"
            constexpr uint64_t kExpectedVersion = 1;

            diskann::NodeIdNvmeWriter nvme_reader(_nvme_dev_path, _nvme_nsid);

            std::unique_ptr<char[], decltype(&std::free)> meta_buf(
                static_cast<char *>(std::aligned_alloc(diskann::NodeIdNvmeWriter::kPageSize,
                                                    kMetaNblks * diskann::NodeIdNvmeWriter::kPageSize)),
                &std::free);

            if (!meta_buf)
            {
                throw diskann::ANNException("aligned_alloc failed for meta_buf", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
            }

            std::memset(meta_buf.get(), 0, kMetaNblks * diskann::NodeIdNvmeWriter::kPageSize);
            nvme_reader.read_node(kMetaNodeId, meta_buf.get(), kMetaNblks);

            if (sizeof(diskann::NodeDiskMeta) > kMetaNblks * diskann::NodeIdNvmeWriter::kPageSize)
            {
                throw diskann::ANNException("NodeDiskMeta does not fit in reserved meta node", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
            }

            const diskann::NodeDiskMeta *meta =
                reinterpret_cast<const diskann::NodeDiskMeta *>(meta_buf.get());

            if (meta->magic != kExpectedMagic)
            {
                std::stringstream ss;
                ss << "Invalid NodeID SSD meta magic: " << meta->magic;
                throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            if (meta->version != kExpectedVersion)
            {
                std::stringstream ss;
                ss << "Unsupported NodeID SSD meta version: " << meta->version;
                throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            uint64_t disk_nnodes = meta->npts;
            uint64_t disk_ndims = meta->ndims;

            if (disk_nnodes != _num_points)
            {
                diskann::cout << "Mismatch in #points for compressed data file and NodeID SSD meta: "
                            << disk_nnodes << " vs " << _num_points << std::endl;
                return -1;
            }

            medoid_id_on_file = static_cast<size_t>(meta->medoid);
            _max_node_len = static_cast<size_t>(meta->max_node_len);

             
            _nnodes_per_sector = 0;

            _max_degree = static_cast<uint32_t>(meta->width);
            if (_max_degree > defaults::MAX_GRAPH_DEGREE)
            {
                std::stringstream stream;
                stream << "Error loading index. Ensure that max graph degree (R) does not exceed "
                    << defaults::MAX_GRAPH_DEGREE << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            this->_num_frozen_points = meta->vamana_frozen_num;
            if (this->_num_frozen_points == 1)
                this->_frozen_location = meta->vamana_frozen_loc;

            this->_reorder_data_exists = (meta->append_reorder_data != 0);
            if (this->_reorder_data_exists)
            {
                if (this->_use_disk_index_pq == false)
                {
                    throw ANNException("Reordering is designed for use with disk PQ compression option",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
                }

                this->_reorder_data_start_sector = meta->reorder_nodeid_base;   
                this->_ndims_reorder_vecs = meta->reorder_ndims;
                this->_nvecs_per_sector = 1;

                _reorder_nodeid_base = meta->reorder_nodeid_base;
                _reorder_nblks = meta->reorder_nblks;
            }

            _data_nodeid_base = meta->data_nodeid_base;
            _node_nblks = meta->node_nblks;
            _node_vec_bytes = meta->node_vec_bytes;

            if (_node_vec_bytes != static_cast<uint64_t>(_data_dim) * sizeof(T))
            {
                std::stringstream ss;
                ss << "NodeID meta node_vec_bytes mismatch, got " << _node_vec_bytes
                << ", expected " << (static_cast<uint64_t>(_data_dim) * sizeof(T));
                throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            _nodeid_reader.reset(new diskann::NodeIdNvmeWriter(_nvme_dev_path, _nvme_nsid));

            diskann::cout << "NodeID SSD Meta-data: ";
            diskann::cout << "data_nodeid_base: " << _data_nodeid_base;
            diskann::cout << ", node_nblks: " << _node_nblks;
            diskann::cout << ", max node len (bytes): " << _max_node_len;
            diskann::cout << ", max node degree: " << _max_degree << std::endl;

            this->setup_thread_data(num_threads);
            this->_max_nthreads = num_threads;
        }
        else
        {
    #ifdef EXEC_ENV_OLS
            reader->open(_disk_index_file);
            this->setup_thread_data(num_threads);
            this->_max_nthreads = num_threads;

            bytes = getHeaderBytes();
            ContentBuf buf(bytes, HEADER_SIZE);
            std::basic_istream<char> index_metadata_stream(&buf);
    #else
            index_metadata.open(_disk_index_file, std::ios::binary);
            std::basic_istream<char> &index_metadata_stream = index_metadata;
    #endif

            uint32_t nr, nc;
            READ_U32(index_metadata_stream, nr);
            READ_U32(index_metadata_stream, nc);

            uint64_t disk_nnodes;
            uint64_t disk_ndims;
            READ_U64(index_metadata_stream, disk_nnodes);
            READ_U64(index_metadata_stream, disk_ndims);

            if (disk_nnodes != _num_points)
            {
                diskann::cout << "Mismatch in #points for compressed data file and disk index file: "
                            << disk_nnodes << " vs " << _num_points << std::endl;
                return -1;
            }

            READ_U64(index_metadata_stream, medoid_id_on_file);
            READ_U64(index_metadata_stream, _max_node_len);
            READ_U64(index_metadata_stream, _nnodes_per_sector);
            _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

            if (_max_degree > defaults::MAX_GRAPH_DEGREE)
            {
                std::stringstream stream;
                stream << "Error loading index. Ensure that max graph degree (R) does not exceed "
                    << defaults::MAX_GRAPH_DEGREE << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            READ_U64(index_metadata_stream, this->_num_frozen_points);
            uint64_t file_frozen_id;
            READ_U64(index_metadata_stream, file_frozen_id);
            if (this->_num_frozen_points == 1)
                this->_frozen_location = file_frozen_id;
            if (this->_num_frozen_points == 1)
            {
                diskann::cout << "Detected frozen point in index at location " << this->_frozen_location
                            << ". Will not output it at search time." << std::endl;
            }

            READ_U64(index_metadata_stream, this->_reorder_data_exists);
            if (this->_reorder_data_exists)
            {
                if (this->_use_disk_index_pq == false)
                {
                    throw ANNException("Reordering is designed for used with disk PQ compression option",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
                }
                READ_U64(index_metadata_stream, this->_reorder_data_start_sector);
                READ_U64(index_metadata_stream, this->_ndims_reorder_vecs);
                READ_U64(index_metadata_stream, this->_nvecs_per_sector);
            }

            diskann::cout << "Disk-Index File Meta-data: ";
            diskann::cout << "# nodes per sector: " << _nnodes_per_sector;
            diskann::cout << ", max node len (bytes): " << _max_node_len;
            diskann::cout << ", max node degree: " << _max_degree << std::endl;

    #ifdef EXEC_ENV_OLS
            delete[] bytes;
    #else
            index_metadata.close();
    #endif
        }

    #ifndef EXEC_ENV_OLS
        if (!_use_nodeid_layout)
        {
            std::string index_fname(_disk_index_file);
            reader->open(index_fname);
            this->setup_thread_data(num_threads);
            this->_max_nthreads = num_threads;
        }
    #endif

     
#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, norm_file, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

     
    std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(files, norm_val, dumr, dumc);
#else
    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
#endif
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }
    diskann::cout << "done.." << std::endl;
    return 0;
}
#pragma endregion


#pragma region

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(std::shared_ptr<AlignedFileReader> &reader, IOContext &ctx, size_t size,
                             int &completedIndex)
{
    if ((*ctx.m_pRequests)[0].m_callback)
    {
        bool waitsRemaining = false;
        long completeCount = ctx.m_completeCount;
        do
        {
            for (int i = 0; i < size; i++)
            {
                auto ithStatus = (*ctx.m_pRequestsStatus)[i];
                if (ithStatus == IOContext::Status::READ_SUCCESS)
                {
                    completedIndex = i;
                    return true;
                }
                else if (ithStatus == IOContext::Status::READ_WAIT)
                {
                    waitsRemaining = true;
                }
            }

            // if we didn't find one in READ_SUCCESS, wait for one to complete.
            if (waitsRemaining)
            {
                WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
                // this assumes the knowledge of the reader behavior (implicit
                // contract). need better factoring?
            }
        } while (waitsRemaining);

        completedIndex = -1;
        return false;
    }
    else
    {
        reader->wait(ctx, completedIndex);
        return completedIndex != -1;
    }
}
#endif

 
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search, uint64_t *indices, float *distances, const uint64_t beam_width, const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),use_reorder_data, stats);
}
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,uint64_t *indices, float *distances, const uint64_t beam_width, const bool use_filter, const LabelT &filter_label,const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,std::numeric_limits<uint32_t>::max(), use_reorder_data, stats);
}
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,uint64_t *indices, float *distances, const uint64_t beam_width,const uint32_t io_limit, const bool use_reorder_data,QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,use_reorder_data, stats);
}

#pragma endregion


 


template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    size_t num_cached_nodes = node_list.size();

    // Allocate space for neighborhood cache
     


    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1) * sizeof(uint32_t));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

     


    size_t block_size = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, block_size);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * block_size;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * block_size);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
         
        auto read_status = read_nodes(_nodeid_reader.get(), nodes_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
         
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    diskann::cout << "..done." << std::endl;
}


 













 
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,uint64_t *indices, float *distances, const uint64_t beam_width,const bool use_filter, const LabelT &filter_label, const uint32_t io_limit, const bool use_reorder_data,QueryStats *stats)
{
     
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);      
    auto data = manager.scratch_space();                                     
    IOContext &ctx = data->ctx;                                              
    NodeIdNvmeWriter* thread_nodeid_reader = nullptr;                        
    NodeIdNvmeWriter* agg_nodeid_reader = nullptr;
    if (_use_nodeid_layout)
    {
        thread_nodeid_reader = data->nodeid_reader.get();                    
        agg_nodeid_reader = data->agg_nodeid_reader.get();
        if (!thread_nodeid_reader)
        {
            throw ANNException("Thread-local NodeID reader is not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (!agg_nodeid_reader)
        {
            throw ANNException("Thread-local Agg NodeID reader is not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }
    auto query_scratch = &(data->scratch);                                   
    auto pq_query_scratch = query_scratch->pq_scratch();                     
    query_scratch->reset();                                                  

     
    float query_norm = 0;                                                    
    T *aligned_query_T = query_scratch->aligned_query_T();                   
    float *query_float = pq_query_scratch->aligned_query_float;              
    float *query_rotated = pq_query_scratch->rotated_query;                  

     
    if (metric == diskann::Metric::INNER_PRODUCT || metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);      
    }

     
    T *data_buf = query_scratch->coord_scratch;                              
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);                             

    char *sector_scratch = query_scratch->sector_scratch;                    
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;                
    // nodeid_bytes_per_node=8192, num_sector_per_nodes=2
    const uint64_t nodeid_bytes_per_node = _use_nodeid_layout ? (static_cast<uint64_t>(_node_nblks) * NodeIdNvmeWriter::kPageSize) : 0;      
    uint64_t num_sector_per_nodes = _use_nodeid_layout ? _node_nblks : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);                    

     
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
    {
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

     
    _pq_table.preprocess_query(query_rotated);                                   
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;            
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);                 
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;                
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;      


     
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids, float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
    };
    
     
    Timer query_timer, io_timer, cpu_timer;
    tsl::robin_set<uint64_t> &visited = query_scratch->visited;          
    NeighborPriorityQueue &retset = query_scratch->retset;               
    retset.reserve(l_search);                                            
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;     
    
    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)             
    {
        float cur_expanded_dist = _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m, (uint32_t)_aligned_dim);
        if (cur_expanded_dist < best_dist)
        {
            best_medoid = _medoids[cur_m];
            best_dist = cur_expanded_dist;
        }
    }
    
    compute_dists(&best_medoid, 1, dist_scratch);                        
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));               
    visited.insert(best_medoid);                                         

    uint32_t cmps = 0;                                                   
    uint32_t hops = 0;                                                   
    uint32_t num_ios = 0;                                                


     
     
    std::vector<uint32_t> frontier;                                      
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;            
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;     
    cached_nhoods.reserve(2 * beam_width);

     
    std::vector<std::pair<uint32_t, NodeIdNvmeWriter::AggCachedNode>> agg_cached_nodes;
    agg_cached_nodes.reserve(2 * beam_width);

    std::vector<NodeIdNvmeWriter::NodeReadReq> nodeid_reqs;             // 
    nodeid_reqs.reserve(2 * beam_width);
    std::vector<uint8_t> read_ok;
    read_ok.reserve(2 * beam_width);


    auto issue_aggregation_prefetch = [&](NeighborPriorityQueue &cur_retset)
    {
        if (!_use_nodeid_layout || !agg_nodeid_reader) {
            return;
        }

        const uint32_t prefetch_num = 8;

        NeighborPriorityQueue tmp = cur_retset;
        std::vector<uint64_t> nodeids;
        tsl::robin_set<uint32_t> selected;

        while (tmp.has_unexpanded_node() && nodeids.size() < prefetch_num)
        {
            auto cand = tmp.closest_unexpanded();
            uint32_t logical_id = cand.id;

            if (selected.find(logical_id) != selected.end()) {
                continue;
            }

            if (_nhood_cache.find(logical_id) != _nhood_cache.end()) {
                continue;
            }

            if (agg_nodeid_reader->has_cached_agg_node(logical_id)) {
                continue;
            }

            selected.insert(logical_id);

            uint64_t physical_id =
                static_cast<uint64_t>(_data_nodeid_base) +
                static_cast<uint64_t>(logical_id);

            nodeids.push_back(physical_id);
        }

        bool submitted = agg_nodeid_reader->submit_aggregation_prefetch_batch_async(nodeids);
        if (!submitted) {
            return;
        }
    };


     
    while (retset.has_unexpanded_node() && num_ios < io_limit)           
    {
        std::vector<NodeIdNvmeWriter::AggPrefetchNode> finished_agg_nodes;
        if (_use_nodeid_layout && agg_nodeid_reader)
        {
            finished_agg_nodes.clear();

            bool ok = agg_nodeid_reader->wait_aggregation_prefetch_batch(finished_agg_nodes);
            if (ok)
            {
                for (const auto &n : finished_agg_nodes)
                {
                    uint32_t logical_id = static_cast<uint32_t>(static_cast<uint64_t>(n.node_id) - static_cast<uint64_t>(_data_nodeid_base));
                    NodeIdNvmeWriter::AggCachedNode cached;
                    cached.vec.resize(this->_data_dim);
                    std::memcpy(cached.vec.data(), n.vector, sizeof(int8_t) * this->_data_dim);

                    cached.degree = std::min<uint32_t>(n.degree, NodeIdNvmeWriter::DISKANN_MAX_DEGREE_FOR_AGG);
                    cached.neighbors.resize(cached.degree);
                    if (cached.degree > 0)
                    {
                        std::memcpy(cached.neighbors.data(), n.neighbors, sizeof(uint32_t) * cached.degree);
                    }
                    agg_nodeid_reader->put_cached_agg_node(logical_id, cached);
                }
            }
        }
        
        
        bool did_ssd_read = false;
        frontier.clear();                                                
        frontier_nhoods.clear();                                         
        cached_nhoods.clear();                                           
        agg_cached_nodes.clear();                                      
        sector_scratch_idx = 0;                                         
        uint32_t num_seen = 0;                                          
        
         
         
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();                      
            num_seen++;                                                  
            auto iter = _nhood_cache.find(nbr.id);                       
            if (iter != _nhood_cache.end())                              
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));   
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                NodeIdNvmeWriter::AggCachedNode cached;                          

                // std::cout << "[LOOKUP] key=" << nbr.id << std::endl;
                if (agg_nodeid_reader->get_cached_agg_node(nbr.id, cached))   
                {
                    // std::cout << "[AGG CACHE HIT] " << nbr.id << std::endl;
                    agg_cached_nodes.emplace_back(nbr.id, std::move(cached));    
                    if (stats != nullptr)
                    {
                        stats->n_cache_hits++;
                    }
                }
                else
                {
                    frontier.push_back(nbr.id);
                }
            }
            if (this->_count_visited_nodes)                              
            {
                reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
            }
        }

         
        if (!frontier.empty())
        {
            did_ssd_read = true;
            if (stats != nullptr)
                stats->n_hops++;

            io_timer.reset();
            if (!thread_nodeid_reader)
            {
                throw ANNException("NodeID reader is not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            nodeid_reqs.clear();
            if (nodeid_reqs.capacity() < frontier.size()) {      
                nodeid_reqs.reserve(frontier.size());
            }
            
             
            for (uint64_t i = 0; i < frontier.size(); i++)      
            {
                auto id = frontier[i];                       
                std::pair<uint32_t, char *> fnhood;          
                fnhood.first = id;                           
                fnhood.second = sector_scratch + sector_scratch_idx * nodeid_bytes_per_node;     
                sector_scratch_idx++;                        
                frontier_nhoods.push_back(fnhood);           

                NodeIdNvmeWriter::NodeReadReq req;           
                req.node_id = static_cast<uint32_t>(_data_nodeid_base + id);     
                // std::cout << "[SSD READ] logical=" << id << " physical=" << req.node_id << std::endl;
                req.buf = fnhood.second;                         
                req.nblks = static_cast<uint32_t>(_node_nblks);
                req.data_len = static_cast<uint32_t>(_max_node_len);   // 4612B
                req.slot_id = 0;
                nodeid_reqs.push_back(req);

                // std::cout << "Request node_id: " << req.node_id << std::endl;
                if (stats != nullptr)
                {
                    stats->n_4k += static_cast<uint32_t>(_node_nblks);   
                    stats->n_ios++;                                      
                }
                num_ios++;
            }


            // trace_mark("DISKANN_READ_START");
            auto read_status = thread_nodeid_reader->read_nodes(nodeid_reqs);        
            // trace_mark("DISKANN_READ_RETURN");
            

             
            std::vector<std::pair<uint32_t, char *>> ok_frontier_nhoods;      
            size_t write_idx = 0;
            for (size_t read_idx = 0; read_idx < frontier_nhoods.size(); ++read_idx)
            {
                if (read_status[read_idx])
                {
                    frontier_nhoods[write_idx++] = frontier_nhoods[read_idx];
                }
            }
            frontier_nhoods.resize(write_idx);
        
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }
        
         
        for (auto &cached_nhood : cached_nhoods)
        {
            auto global_cache_iter = _coord_cache.find(cached_nhood.first);  
            T *node_fp_coords_copy = global_cache_iter->second;
            float cur_expanded_dist;
            if (!_use_disk_index_pq)                                         
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)node_fp_coords_copy);
            }
            
            full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));    
            
            uint64_t nnbrs = cached_nhood.second.first;          
            uint32_t *node_nbrs = cached_nhood.second.second;

            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);       
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)                                   
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())  
                        continue;
                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];                                
                    Neighbor nn(id, dist);
                    retset.insert(nn);                                           
                }
            }
        }

         
        for (auto &agg_cached_node : agg_cached_nodes)
        {
            uint32_t expanded_node_id = agg_cached_node.first;
            NodeIdNvmeWriter::AggCachedNode &cached = agg_cached_node.second;

            if (cached.vec.empty())
            {
                continue;
            }
            float cur_expanded_dist = _dist_cmp->compare(aligned_query_T, reinterpret_cast<T *>(cached.vec.data()), static_cast<uint32_t>(this->_data_dim));
            full_retset.push_back(Neighbor(expanded_node_id, cur_expanded_dist));
            
            uint64_t nnbrs = cached.degree;
            uint32_t *node_nbrs = cached.neighbors.data();
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += static_cast<uint32_t>(nnbrs);
                stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
            }
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                    {
                        continue;
                    }
                    if (use_filter && !(point_has_label(id, filter_label)) && (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                    {
                        continue;
                    }
                    cmps++;
                    float dist = dist_scratch[m];
                    retset.insert(Neighbor(id, dist));
                }
            }
        }

         
        for (auto &frontier_nhood : frontier_nhoods)
        {
            char *node_disk_buf = _use_nodeid_layout ? frontier_nhood.second : offset_to_node(frontier_nhood.second, frontier_nhood.first);     
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);                                
            
            uint64_t nnbrs = (uint64_t)(*node_buf);                                                  
            if (nnbrs > 64) {
                std::cerr << "[BAD DEGREE][SSD]"
                        << " node=" << frontier_nhood.first
                        << " degree=" << nnbrs
                        << " ptr=" << (void*)node_buf
                        << std::endl;
                abort();
            }
            
            
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);                                
            const uint64_t node_vec_bytes = _use_nodeid_layout ? _node_vec_bytes : _disk_bytes_per_point;
            memcpy(data_buf, node_fp_coords, node_vec_bytes);                               
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }

            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));                
            uint32_t *node_nbrs = (node_buf + 1);                                                    
            for (uint64_t m = 0; m < nnbrs; ++m) {
                if (node_nbrs[m] >= this->_num_points) {
                    std::cerr << "[BAD NBR][SSD]"
                            << " node=" << frontier_nhood.first
                            << " m=" << m
                            << " nbr=" << node_nbrs[m]
                            << " num_points=" << this->_num_points
                            << std::endl;
                    abort();
                }
            }
            // std::cout << "[SSD EXPAND] node=" << frontier_nhood.first << " degree=" << nnbrs << " nbr0=" << node_nbrs[0] << " nbr1=" << node_nbrs[1] << std::endl

            cpu_timer.reset();                                                                       
            compute_dists(node_nbrs, nnbrs, dist_scratch);                                           
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
            cpu_timer.reset();                                                                       
            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)                                                     
            {
                uint32_t id = node_nbrs[m];                                                          
                if (visited.insert(id).second)                                                       
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())                      
                        continue;
                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))    
                        continue;
                    cmps++;                                                                          
                    float dist = dist_scratch[m];                                                    
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }
                    Neighbor nn(id, dist);                                                           
                    retset.insert(nn);                                                               
                }
            }
            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }
         
        if(did_ssd_read)
        {
            io_timer.reset();
            issue_aggregation_prefetch(retset);
        
            if(stats!=nullptr)
            {
                stats->n_ios++;
                stats->io_us += (float)io_timer.elapsed();
            }
        }
        hops++;
    }
    // consume_aggregation_prefetch();
     
    #pragma region
    std::sort(full_retset.begin(), full_retset.end());
    if (use_reorder_data)
    {
        if (!(this->_reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does not exist in index file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
        {
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
                              full_retset.end());
        }

        io_timer.reset();

        if (_use_nodeid_layout)
        {
            if (!_nodeid_reader)
            {
                throw ANNException("NodeID reader is not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
            }

            const uint64_t reorder_bytes_per_vec =
                static_cast<uint64_t>(_reorder_nblks) * NodeIdNvmeWriter::kPageSize;

            std::vector<NodeIdNvmeWriter::NodeReadReq> reorder_reqs;
            reorder_reqs.reserve(full_retset.size());

            for (size_t i = 0; i < full_retset.size(); ++i)
            {
                NodeIdNvmeWriter::NodeReadReq req;
                req.node_id = static_cast<uint32_t>(_reorder_nodeid_base + full_retset[i].id);
                req.buf = sector_scratch + i * reorder_bytes_per_vec;
                req.nblks = static_cast<uint32_t>(_reorder_nblks);
                req.data_len = static_cast<uint32_t>(reorder_bytes_per_vec);
                reorder_reqs.push_back(req);

                if (stats != nullptr)
                {
                    stats->n_4k += static_cast<uint32_t>(_reorder_nblks);
                    stats->n_ios++;
                }
            }

            auto reorder_status = thread_nodeid_reader->read_nodes(reorder_reqs);

            if (stats != nullptr)
            {
                stats->io_us += io_timer.elapsed();
            }

            for (size_t i = 0; i < full_retset.size(); ++i)
            {
                if (!reorder_status[i])
                {
                    throw ANNException("NodeID reorder vector read failed",
                                       -1, __FUNCSIG__, __FILE__, __LINE__);
                }

                full_retset[i].distance =
                    _dist_cmp->compare(aligned_query_T,
                                       reinterpret_cast<T *>(sector_scratch + i * reorder_bytes_per_vec),
                                       (uint32_t)this->_data_dim);
            }
        }
        else
        {
            std::vector<AlignedRead> vec_read_reqs;

            for (size_t i = 0; i < full_retset.size(); ++i)
            {
                vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * defaults::SECTOR_LEN,
                                           defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
            }

#ifdef USE_BING_INFRA
            reader->read(vec_read_reqs, ctx, true);
#else
            reader->read(vec_read_reqs, ctx);
#endif
            if (stats != nullptr)
            {
                stats->io_us += io_timer.elapsed();
            }

            for (size_t i = 0; i < full_retset.size(); ++i)
            {
                auto id = full_retset[i].id;
                auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
                full_retset[i].distance =
                    _dist_cmp->compare(aligned_query_T, (T *)location, (uint32_t)this->_data_dim);
            }
        }

        std::sort(full_retset.begin(), full_retset.end());
    }
    #pragma endregion
    
     
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        auto key = (uint32_t)indices[i];
        if (_dummy_pts.find(key) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[key];
        }

        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    distances[i] *= (_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif
     
    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
    }
}

#pragma region
// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                               const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                               std::vector<float> &distances, const uint64_t min_beam_width,
                                               QueryStats *stats)
{
    uint32_t res_count = 0;

    bool stop_flag = false;

    uint32_t l_search = (uint32_t)min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false, stats);
        for (uint32_t i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (uint32_t)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
#pragma endregion