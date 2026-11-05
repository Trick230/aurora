#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

// The references and ranges draws are made of, with no graphics API in them.
namespace aurora {
using HashType = std::conditional_t<sizeof(void*) == 4, uint32_t, uint64_t>;
} // namespace aurora

namespace aurora::gfx {
using BindGroupRef = HashType;
using PipelineRef = HashType;
using SamplerRef = HashType;
using ShaderRef = HashType;
struct Range {
  uint32_t offset = 0;
  uint32_t size = 0;

  bool operator==(const Range& rhs) const { return memcmp(this, &rhs, sizeof(*this)) == 0; }
  bool operator!=(const Range& rhs) const { return !(*this == rhs); }
};
} // namespace aurora::gfx
