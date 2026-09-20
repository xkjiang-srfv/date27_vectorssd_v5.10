#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

class RawBlockWriter {
public:
  RawBlockWriter(const std::string& path, size_t align_bytes, bool use_direct)
      : _fd(-1), _off(0), _align(align_bytes) {

    int flags = O_WRONLY;
     
     
    if (use_direct) flags |= O_DIRECT;
     
    // flags |= O_DSYNC;

    _fd = ::open(path.c_str(), flags);
    if (_fd < 0) {
      throw std::runtime_error("open(" + path + ") failed: " + std::string(std::strerror(errno)));
    }
  }

  ~RawBlockWriter() {
    if (_fd >= 0) ::close(_fd);
  }

   
  void write(const void* buf, size_t nbytes) {
    write_at(_off, buf, nbytes);
    _off += nbytes;
  }

   
  void write_at(uint64_t offset, const void* buf, size_t nbytes) {
     
    if ((offset % _align) != 0 || (nbytes % _align) != 0) {
      throw std::runtime_error("unaligned write: offset or size not multiple of align");
    }
     
    if ((reinterpret_cast<uintptr_t>(buf) % _align) != 0) {
      throw std::runtime_error("unaligned buffer address for direct IO");
    }

    const char* p = reinterpret_cast<const char*>(buf);
    size_t left = nbytes;
    while (left > 0) {
      ssize_t w = ::pwrite(_fd, p, left, static_cast<off_t>(offset));
      if (w < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("pwrite failed: " + std::string(std::strerror(errno)));
      }
      left -= static_cast<size_t>(w);
      p += w;
      offset += static_cast<uint64_t>(w);
    }
  }

  void close_and_sync() {
    ::fsync(_fd);
    ::close(_fd);
    _fd = -1;
  }

  uint64_t tell() const { return _off; }

private:
  int _fd;
  uint64_t _off;
  size_t _align;
};