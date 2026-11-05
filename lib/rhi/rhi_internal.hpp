#pragma once

#include "rhi.hpp"

#include <algorithm>
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace aurora::rhi {

std::unique_ptr<Device> create_d3d12_device(const DeviceOptions& options, std::string* error);
std::unique_ptr<Device> create_vulkan_device(const DeviceOptions& options, std::string* error);

extern std::atomic<int64_t> g_validation_errors;
void validation_active();

// What was destroyed, released once the GPU is done with it. A list being
// recorded may still use it, and other lists may be submitted before that one
// is: an object goes once every list open when it was destroyed was submitted,
// and the GPU got past both those submits and the next one.
class RetireQueue {
public:
  // A list began recording; one begun again without a submit gave up the last recording.
  void opened(const void* list) {
    std::lock_guard lock{m_mutex};
    if (const auto it = m_open.find(list); it != m_open.end()) {
      for (const auto& entry : it->second)
        --entry->waiting;
      it->second.clear();
    } else {
      m_open.emplace(list, std::vector<std::shared_ptr<Entry>>{});
    }
  }
  // A list went away, submitted or not.
  void closed(const void* list) {
    std::lock_guard lock{m_mutex};
    if (const auto it = m_open.find(list); it != m_open.end()) {
      for (const auto& entry : it->second)
        --entry->waiting;
      m_open.erase(it);
    }
  }
  void submitted(const void* list, uint64_t value) {
    std::lock_guard lock{m_mutex};
    if (const auto it = m_open.find(list); it != m_open.end()) {
      for (const auto& entry : it->second) {
        entry->value = std::max(entry->value, value);
        --entry->waiting;
      }
      m_open.erase(it);
    }
  }
  // next: what the next submit will signal.
  void retire(uint64_t next, std::function<void()> release) {
    std::lock_guard lock{m_mutex};
    auto entry = std::make_shared<Entry>(Entry{next, static_cast<uint32_t>(m_open.size()), std::move(release)});
    for (auto& [list, entries] : m_open)
      entries.push_back(entry);
    m_entries.push_back(std::move(entry));
  }
  // Releases, outside the lock, what the GPU is done with (all of it with everything).
  void collect(uint64_t done, bool everything = false) {
    std::vector<std::function<void()>> ready;
    {
      std::lock_guard lock{m_mutex};
      while (!m_entries.empty() &&
             (everything || (m_entries.front()->waiting == 0 && m_entries.front()->value <= done))) {
        ready.push_back(std::move(m_entries.front()->release));
        m_entries.pop_front();
      }
    }
    for (auto& release : ready)
      release();
  }

private:
  struct Entry {
    uint64_t value;
    uint32_t waiting;
    std::function<void()> release;
  };
  std::mutex m_mutex;
  std::deque<std::shared_ptr<Entry>> m_entries;
  std::unordered_map<const void*, std::vector<std::shared_ptr<Entry>>> m_open;
};

inline constexpr uint32_t MipExtent(uint32_t size, uint32_t mip) { return size >> mip ? size >> mip : 1; }

} // namespace aurora::rhi
