#pragma once

#include "rhi.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
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

// The GPU time of lists begun while timing is on: each claims a pair of timestamps
// in a ring, read once its submit is done. A full ring skips timing a list rather
// than wait. Lists may begin on another thread than the one that submits, so the
// pairs' states are atomics; none of this runs while timing is off.
class GpuTimer {
public:
  static constexpr uint32_t Pairs = 64;

  void set_enabled(bool on) { m_enabled.store(on, std::memory_order_relaxed); }
  bool enabled() const { return m_enabled.load(std::memory_order_relaxed); }
  // A free pair for a list beginning, or -1. A list begun again keeps its pair.
  int32_t claim(int32_t current) {
    if (current >= 0)
      return current;
    if (!enabled())
      return -1;
    const uint32_t pair = m_next.fetch_add(1, std::memory_order_relaxed) % Pairs;
    uint64_t expected = Free;
    if (!m_state[pair].compare_exchange_strong(expected, Open, std::memory_order_acq_rel))
      return -1;
    return static_cast<int32_t>(pair);
  }
  // A list went away without a submit.
  void release(int32_t pair) {
    if (pair >= 0)
      m_state[pair].store(Free, std::memory_order_release);
  }
  void submitted(int32_t pair, uint64_t ticket) {
    m_pending.fetch_add(1, std::memory_order_relaxed);
    m_state[pair].store(ticket, std::memory_order_release);
  }
  // From the submitting thread: adds the time of the pairs whose submit is done;
  // read(pair) gives it in nanoseconds.
  template <typename Read>
  void collect(uint64_t done, Read&& read) {
    if (m_pending.load(std::memory_order_relaxed) == 0)
      return;
    uint64_t ns = 0;
    for (uint32_t pair = 0; pair < Pairs; ++pair) {
      const uint64_t state = m_state[pair].load(std::memory_order_acquire);
      if (state == Free || state == Open || state > done)
        continue;
      ns += read(pair);
      m_state[pair].store(Free, std::memory_order_release);
      m_pending.fetch_sub(1, std::memory_order_relaxed);
    }
    if (ns != 0)
      m_ns.fetch_add(ns, std::memory_order_relaxed);
  }
  uint64_t take() { return m_ns.exchange(0, std::memory_order_relaxed); }

private:
  static constexpr uint64_t Free = 0;
  static constexpr uint64_t Open = UINT64_MAX;
  std::atomic<bool> m_enabled{false};
  std::atomic<uint32_t> m_next{0};
  std::atomic<uint32_t> m_pending{0};
  std::atomic<uint64_t> m_ns{0};
  std::array<std::atomic<uint64_t>, Pairs> m_state{};
};

inline constexpr uint32_t MipExtent(uint32_t size, uint32_t mip) { return size >> mip ? size >> mip : 1; }

} // namespace aurora::rhi
