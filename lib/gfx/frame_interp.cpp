#include "frame_interp.hpp"

#include "render_worker.hpp"
#include "../gx/gx.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../window.hpp"

#include <SDL3/SDL_video.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif


namespace aurora::gfx {
const wgpu::BindGroup* interp::g_vertexUniformBindGroup = &g_uniformBindGroup;
}

namespace aurora::gfx::interp {
namespace {
Module Log("aurora::gfx::interp");

using Clock = std::chrono::steady_clock;
using webgpu::g_device;
using webgpu::g_queue;

// The matrices of a GX uniform block, in floats (gx/shader_info.cpp
// build_uniform): the projection, the position matrices, the texture matrices
// and the normal matrices. Texture matrices blend only while they move a
// little, since a scrolling texture's offset wraps.
constexpr uint32_t PosMtx = sizeof(Mat4x4<float>) / 4;
constexpr uint32_t TexMtx = PosMtx + gx::MaxPnMtx * sizeof(Mat3x4<float>) / 4;
constexpr uint32_t NrmMtx = TexMtx + gx::MaxTexMtx * sizeof(Mat3x4<float>) / 4;
constexpr uint32_t BlendFloats = NrmMtx + gx::MaxPnMtx * sizeof(Mat3x4<float>) / 4;
constexpr uint32_t BlendBytes = BlendFloats * 4;
static_assert(PosMtx == 16 && TexMtx == 136 && NrmMtx == 256 && BlendFloats == 376);
constexpr uint32_t MtxFloats = sizeof(Mat3x4<float>) / 4;
constexpr float MaxTexDelta = 0.25f;
// Further on, a block may hold lights (each a position, a direction, a color
// and attenuations) and the post-transform texture matrices.
constexpr uint32_t LightFloats = sizeof(gx::Light) / 4;
constexpr uint32_t LightsFloats = ::GX::MaxLights * LightFloats;
constexpr uint32_t PostFloats = gx::MaxPTTexMtx * MtxFloats;
static_assert(LightFloats == 20);

// Matrix n of a block: 0 the projection, then the position, texture and
// normal matrices.
constexpr uint32_t MatrixCount = 1 + 2 * gx::MaxPnMtx + gx::MaxTexMtx;
static_assert(MatrixCount <= 32);
constexpr uint32_t matrix_start(uint32_t n) { return n == 0 ? 0 : PosMtx + (n - 1) * MtxFloats; }
constexpr uint32_t matrix_floats(uint32_t n) { return n == 0 ? PosMtx : MtxFloats; }
constexpr bool is_texture_matrix(uint32_t n) { return n > gx::MaxPnMtx && n <= gx::MaxPnMtx + gx::MaxTexMtx; }
static_assert(matrix_start(gx::MaxPnMtx + 1) == TexMtx && matrix_start(gx::MaxPnMtx + gx::MaxTexMtx + 1) == NrmMtx);

struct Frame {
  ByteBuffer uniforms;
  std::vector<DrawRecord> draws;
  std::vector<ViewMatrix> cameras;
  ReplayFn replay;
  EndFrameCallback present;
  SubmitFn submit;
  Clock::time_point arrival;
};

static_assert(MatrixSlots == gx::MaxPnMtx);

// What a translation follows from one frame to the next.
enum class Follow : uint8_t {
  Eye,   // the eye: as it is
  World, // the world: the whole camera move
  Turn,  // the camera's turn alone
};

struct Blend {
  uint32_t offset;     // float index of the matrices in the newer frame
  uint32_t prevOffset; // and of their match in the older one
  uint32_t matrices;   // bit n: matrix n differs
  // The older frame's slot paired with each position, then normal, matrix.
  std::array<uint8_t, 2 * gx::MaxPnMtx> prevSlot;
  // prevOffset is into g_older. With these set, the rotation (translation)
  // of its position and normal matrices is in the newer frame's eye space,
  // and that part of the blend is carried back along the camera move.
  // Billboards (trees, sprites) keep their place in the world but face the
  // camera: only their translation is in camera space. A backdrop (sky,
  // distant scenery) is drawn around the eye: its translation only turns
  // with the camera.
  bool cameraRotation = false;
  Follow cameraTranslation = Follow::Eye;
  uint16_t camera = 0; // the camera move it follows, in g_moves
  // Where the block's lights and post matrices are (0: none), and the floats
  // of it g_older holds.
  uint16_t lights = 0;
  uint16_t post = 0;
  uint32_t extent = BlendFloats;
  // Bit i: light (post matrix) i differs, and it stands still in the world:
  // it is then in the newer eye space in g_older and carried back like the
  // matrices. Others are as they are.
  uint8_t lightBlend = 0;
  uint8_t lightWorld = 0;
  uint32_t postBlend = 0;
  uint32_t postWorld = 0;
};

constexpr uint32_t FirstNrm = 1 + gx::MaxPnMtx + gx::MaxTexMtx;

// Where matrix n's older counterpart starts within the older block.
uint32_t prev_start(const Blend& blend, uint32_t n) {
  if (n >= FirstNrm) {
    return matrix_start(FirstNrm + blend.prevSlot[gx::MaxPnMtx + n - FirstNrm]);
  }
  if (n >= 1 && n <= gx::MaxPnMtx) {
    return matrix_start(1 + blend.prevSlot[n - 1]);
  }
  return matrix_start(n);
}

constexpr uint8_t NoSlot = 0xFF;

// The older draw's slot holding what the newer one loaded into a slot: the
// one loaded from the same index, or for inline loads the same slot if that
// was inline too.
uint8_t pair_slot(uint32_t source, uint32_t slot, const DrawRecord& other, uint32_t MatrixSources::* field) {
  if (source == 0) {
    return other.sources[slot].*field == 0 ? static_cast<uint8_t>(slot) : NoSlot;
  }
  for (uint32_t j = 0; j < gx::MaxPnMtx; ++j) {
    if (other.sources[j].*field == source) {
      return static_cast<uint8_t>(j);
    }
  }
  return NoSlot;
}

bool g_enabled = false;
bool g_vsync = true;
Clock::duration g_interval{};
wgpu::Buffer g_buffer;
wgpu::BindGroup g_bindGroup;
std::atomic_bool g_replaying = false;

// Render worker only.
Frame g_prev;
Frame g_cur;
// The newest frame, held back (unsubmitted) while g_cur is still shown.
Frame g_next;
std::vector<Blend> g_blends;
// Older matrices brought into the newer frame's eye space, slot for slot.
std::vector<float> g_older;
ByteBuffer g_scratch;
uint32_t g_dirtyBegin = 0;
uint32_t g_dirtyEnd = 0;
bool g_uploaded = false;
absl::flat_hash_set<uint32_t> g_blocks;
Clock::time_point g_base{};
Clock::time_point g_lastArrival{};
Clock::time_point g_shown{};
Clock::time_point g_lastPresent{};
Clock::duration g_period = std::chrono::nanoseconds{1'000'000'000 / 30};

// Pacer handshake.
std::mutex g_mutex;
std::condition_variable g_cv;
std::thread g_pacer;
bool g_stop = false;
bool g_pending = false; // the newest frame has not been shown at t = 1 yet
bool g_ticking = false;

constexpr uint32_t NoMatch = UINT32_MAX;

// Whether texture matrix a to b scrolls across a wrap (its offset jumps by
// about a whole texture), which a blend would sweep across the texture. Its
// rotation and scale, however large the change (lighting, reflections on
// moving geometry), blend.
bool wraps(const float* a, const float* b) {
  return !(std::abs(b[3] - a[3]) < MaxTexDelta && std::abs(b[7] - a[7]) < MaxTexDelta &&
           std::abs(b[11] - a[11]) < MaxTexDelta);
}

// Affine 3x4 matrices, row-major like GX position matrices.
using Affine = std::array<float, 12>;
constexpr Affine Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

bool invert(const float* m, Affine& out) {
  const float c00 = m[5] * m[10] - m[6] * m[9];
  const float c01 = m[6] * m[8] - m[4] * m[10];
  const float c02 = m[4] * m[9] - m[5] * m[8];
  const float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  if (!(std::abs(det) > 1e-12f)) {
    return false;
  }
  const float inv = 1.f / det;
  const float a[9] = {
      c00 * inv, (m[2] * m[9] - m[1] * m[10]) * inv, (m[1] * m[6] - m[2] * m[5]) * inv,
      c01 * inv, (m[0] * m[10] - m[2] * m[8]) * inv, (m[2] * m[4] - m[0] * m[6]) * inv,
      c02 * inv, (m[1] * m[8] - m[0] * m[9]) * inv, (m[0] * m[5] - m[1] * m[4]) * inv,
  };
  for (int r = 0; r < 3; ++r) {
    out[r * 4 + 0] = a[r * 3 + 0];
    out[r * 4 + 1] = a[r * 3 + 1];
    out[r * 4 + 2] = a[r * 3 + 2];
    out[r * 4 + 3] = -(a[r * 3 + 0] * m[3] + a[r * 3 + 1] * m[7] + a[r * 3 + 2] * m[11]);
  }
  return true;
}

// x, then y.
Affine compose(const float* y, const float* x) {
  Affine out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      out[r * 4 + c] = y[r * 4 + 0] * x[c] + y[r * 4 + 1] * x[4 + c] + y[r * 4 + 2] * x[8 + c] +
                       (c == 3 ? y[r * 4 + 3] : 0.f);
    }
  }
  return out;
}

using Vec3 = std::array<float, 3>;

Vec3 translation(const float* m) { return {m[3], m[7], m[11]}; }

Vec3 transform_point(const Affine& d, const Vec3& p) {
  return {
      d[0] * p[0] + d[1] * p[1] + d[2] * p[2] + d[3],
      d[4] * p[0] + d[5] * p[1] + d[6] * p[2] + d[7],
      d[8] * p[0] + d[9] * p[1] + d[10] * p[2] + d[11],
  };
}

float distance2(const Vec3& a, const Vec3& b) {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  const float dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

float length2(const Vec3& a) { return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; }

// The camera move between the two frames that the draw being matched follows
// (see use_camera): the map from the older frame's eye space to the newer
// one's.
Affine g_camera = Identity;
Affine g_cameraInverse = Identity;
bool g_cameraMoved = false;

// The camera move from the newer frame back to the older one as a screw: a
// turn about an axis through a point, a shift along it, and a scale. Carrying
// a blend back by a fraction of it follows the arc a camera orbiting the
// player takes, where blending matrices would cut across the chord.
struct Screw {
  Vec3 axis;
  float angle;
  Vec3 center;
  Vec3 shift; // along the axis; the whole translation when there is no turn
  float scale;
};
Screw g_screw;
// What the screw misses of the move (the camera move is not quite rigid);
// added in proportion, so the whole screw lands exactly on it.
Affine g_screwError{};

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Screw make_screw(const Affine& m) {
  Screw screw{{0, 0, 1}, 0.f, {0, 0, 0}, translation(m.data()), 1.f};
  const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                    m[2] * (m[4] * m[9] - m[5] * m[8]);
  if (!(det > 0.f)) {
    return screw;
  }
  screw.scale = std::cbrt(det);
  const float k = 1.f / screw.scale;
  const float trace = (m[0] + m[5] + m[10]) * k;
  const float angle = std::acos(std::clamp((trace - 1.f) * 0.5f, -1.f, 1.f));
  const Vec3 axis{(m[9] - m[6]) * k, (m[2] - m[8]) * k, (m[4] - m[1]) * k};
  const float sine = std::sqrt(length2(axis)) * 0.5f;
  if (!(angle > 1e-5f) || !(sine > 1e-6f)) {
    return screw;
  }
  const float inv = 1.f / (2.f * sine);
  screw.axis = {axis[0] * inv, axis[1] * inv, axis[2] * inv};
  screw.angle = angle;
  const Vec3 t = screw.shift;
  const float along = t[0] * screw.axis[0] + t[1] * screw.axis[1] + t[2] * screw.axis[2];
  screw.shift = {screw.axis[0] * along, screw.axis[1] * along, screw.axis[2] * along};
  const Vec3 across{t[0] - screw.shift[0], t[1] - screw.shift[1], t[2] - screw.shift[2]};
  const Vec3 turn = cross(screw.axis, across);
  const float cot = 1.f / std::tan(angle * 0.5f);
  screw.center = {(across[0] + cot * turn[0]) * 0.5f, (across[1] + cot * turn[1]) * 0.5f,
                  (across[2] + cot * turn[2]) * 0.5f};
  return screw;
}

// The fraction s of the screw: s = 0 is no move, s = 1 the whole one.
Affine screw_at(const Screw& screw, float s) {
  const float angle = screw.angle * s;
  const float sine = std::sin(angle);
  const float versine = 1.f - std::cos(angle);
  const float scale = std::pow(screw.scale, s);
  const auto& u = screw.axis;
  // Rodrigues: I + sin K + (1 - cos) K^2, K the cross product with u.
  const float r[9] = {
      1.f - versine * (u[1] * u[1] + u[2] * u[2]), versine * u[0] * u[1] - sine * u[2], versine * u[0] * u[2] + sine * u[1],
      versine * u[0] * u[1] + sine * u[2], 1.f - versine * (u[0] * u[0] + u[2] * u[2]), versine * u[1] * u[2] - sine * u[0],
      versine * u[0] * u[2] - sine * u[1], versine * u[1] * u[2] + sine * u[0], 1.f - versine * (u[0] * u[0] + u[1] * u[1]),
  };
  const auto& c = screw.center;
  Affine out;
  for (int row = 0; row < 3; ++row) {
    const float* rr = r + row * 3;
    out[row * 4 + 0] = rr[0] * scale;
    out[row * 4 + 1] = rr[1] * scale;
    out[row * 4 + 2] = rr[2] * scale;
    out[row * 4 + 3] = c[row] - (rr[0] * c[0] + rr[1] * c[1] + rr[2] * c[2]) * scale + screw.shift[row] * s;
  }
  return out;
}

// A camera move between the two frames, and its screw.
struct CameraMove {
  Affine move = Identity;
  Affine inverse = Identity;
  bool moved = false;
  Screw screw;
  Affine screwError{};
};
// 0: the move estimated from the draws; 1 + n: camera n of both frames, as
// the game gave it.
std::vector<CameraMove> g_moves;

CameraMove make_move(const Affine& move) {
  CameraMove out;
  out.move = move;
  out.moved = move != Identity && invert(move.data(), out.inverse);
  if (!out.moved) {
    out.move = out.inverse = Identity;
    return out;
  }
  out.screw = make_screw(out.inverse);
  const Affine whole = screw_at(out.screw, 1.f);
  for (int q = 0; q < 12; ++q) {
    out.screwError[q] = out.inverse[q] - whole[q];
  }
  return out;
}

// The moves of the two frames' cameras, paired in the order the game gave them.
void exact_cameras() {
  g_moves.resize(1);
  const size_t count = std::min(g_prev.cameras.size(), g_cur.cameras.size());
  for (size_t n = 0; n < count; ++n) {
    Affine inverse;
    if (!invert(g_prev.cameras[n].data(), inverse)) {
      inverse = Identity;
    }
    g_moves.push_back(make_move(compose(g_cur.cameras[n].data(), inverse.data())));
  }
}

// The camera move a draw of the newer frame follows: its own camera's, else
// the estimated one.
uint16_t camera_of(const DrawRecord& draw) { return draw.camera < g_moves.size() ? draw.camera : 0; }

// Makes the move the helpers below use the given one.
void use_camera(uint16_t camera) {
  const auto& move = g_moves[camera];
  g_camera = move.move;
  g_cameraInverse = move.inverse;
  g_cameraMoved = move.moved;
  g_screw = move.screw;
  g_screwError = move.screwError;
}

struct Slot {
  HashType key;
  uint32_t index;
};
struct SlotRange {
  uint32_t begin;
  uint32_t end;
  uint32_t next; // the first slot not yet taken in order
};
struct SlotIndex {
  std::vector<Slot> slots; // older frame's draws, by key, then in draw order
  absl::flat_hash_map<HashType, SlotRange> ranges;
};
SlotIndex g_byKey;
SlotIndex g_byGroup;
std::vector<uint8_t> g_used;
// The older draw paired with each newer one.
std::vector<uint32_t> g_pairs;

// A draw's identity: a hash (its key, or for geometry whose size changes its
// group) and the address its current matrix was loaded from. The game keeps
// each model's matrices in arrays of its own, so this tells instances of a
// model apart, and the parts of one, across frames.
uint64_t identity(const DrawRecord& draw, HashType DrawRecord::* hash) {
  const uint32_t source = draw.current < gx::MaxPnMtx ? draw.sources[draw.current].pos : 0;
  return source == 0 ? 0 : (static_cast<uint64_t>(draw.*hash) * 0x9E3779B97F4A7C15ull) ^ source;
}

// The older frame's draws by identity, chained in draw order for repeats.
struct IdentityIndex {
  HashType DrawRecord::* hash;
  absl::flat_hash_map<uint64_t, uint32_t> first;
  std::vector<uint32_t> next;
};
IdentityIndex g_byKeyIdentity{&DrawRecord::key};
IdentityIndex g_byGroupIdentity{&DrawRecord::group};

void build_identities(IdentityIndex& index, const std::vector<DrawRecord>& draws) {
  index.first.clear();
  index.next.assign(draws.size(), NoMatch);
  for (uint32_t i = static_cast<uint32_t>(draws.size()); i-- > 0;) {
    const uint64_t id = identity(draws[i], index.hash);
    if (id != 0) {
      auto [it, inserted] = index.first.try_emplace(id, i);
      if (!inserted) {
        index.next[i] = it->second;
        it->second = i;
      }
    }
  }
}

void build_index(SlotIndex& index, const std::vector<DrawRecord>& draws, HashType DrawRecord::*field) {
  index.slots.clear();
  for (uint32_t i = 0; i < draws.size(); ++i) {
    index.slots.push_back({draws[i].*field, i});
  }
  std::ranges::sort(index.slots, [](const Slot& a, const Slot& b) { return a.key < b.key || (a.key == b.key && a.index < b.index); });
  index.ranges.clear();
  for (uint32_t begin = 0; begin < index.slots.size();) {
    uint32_t end = begin + 1;
    while (end < index.slots.size() && index.slots[end].key == index.slots[begin].key) {
      ++end;
    }
    index.ranges.emplace(index.slots[begin].key, SlotRange{begin, end, begin});
    begin = end;
  }
}

// The position matrix a draw's vertices use by default.
const float* current_matrix(const float* base, const DrawRecord& draw) {
  return base + draw.offset / 4 + PosMtx + std::min(draw.current, gx::MaxPnMtx - 1) * MtxFloats;
}

absl::flat_hash_map<HashType, uint32_t> g_curCounts;
std::vector<std::pair<uint32_t, uint32_t>> g_samples;

// Draws whose key occurs once in both frames are surely the same object; the
// camera move is the one most of them agree on, as most of a scene stands still.
void estimate_camera() {
  Affine camera = Identity;
  g_curCounts.clear();
  for (const auto& draw : g_cur.draws) {
    ++g_curCounts[draw.key];
  }
  g_samples.clear();
  const auto* prevBase = reinterpret_cast<const float*>(g_prev.uniforms.data());
  const auto* curBase = reinterpret_cast<const float*>(g_cur.uniforms.data());
  for (uint32_t i = 0; i < g_cur.draws.size(); ++i) {
    const auto& draw = g_cur.draws[i];
    const auto it = g_byKey.ranges.find(draw.key);
    if (g_curCounts[draw.key] != 1 || it == g_byKey.ranges.end() || it->second.end - it->second.begin != 1 ||
        draw.offset + BlendBytes > g_cur.uniforms.size()) {
      continue;
    }
    const uint32_t j = g_byKey.slots[it->second.begin].index;
    if (g_prev.draws[j].offset + BlendBytes <= g_prev.uniforms.size()) {
      g_samples.emplace_back(j, i);
    }
  }
  constexpr size_t MaxSamples = 96;
  const size_t step = std::max<size_t>(1, g_samples.size() / MaxSamples);
  uint32_t bestVotes = 0;
  for (size_t k = 0; k < g_samples.size(); k += step) {
    Affine inverse;
    if (!invert(current_matrix(prevBase, g_prev.draws[g_samples[k].first]), inverse)) {
      continue;
    }
    const Affine candidate = compose(current_matrix(curBase, g_cur.draws[g_samples[k].second]), inverse.data());
    uint32_t votes = 0;
    for (size_t l = 0; l < g_samples.size(); l += step) {
      const Vec3 tp = translation(current_matrix(prevBase, g_prev.draws[g_samples[l].first]));
      const Vec3 tc = translation(current_matrix(curBase, g_cur.draws[g_samples[l].second]));
      votes += distance2(transform_point(candidate, tp), tc) < 1e-4f * length2(tc) + 1e-4f;
    }
    if (votes > bestVotes) {
      bestVotes = votes;
      camera = candidate;
    }
  }
  g_moves.resize(1);
  g_moves[0] = make_move(bestVotes < 3 ? Identity : camera);
}

// How far, squared, a draw at target may have moved to pair with the closest
// of its model's instances: 5% of its distance from the eye.
float near_enough(const Vec3& target) { return 0.0025f * length2(target) + 1e-4f; }

// How many of the position matrices draw a uses it loaded from an address,
// and how many of those b loaded from too.
void count_sources(const DrawRecord& a, const DrawRecord& b, uint32_t& loaded, uint32_t& shared) {
  for (uint32_t n = 0; n < gx::MaxPnMtx; ++n) {
    const auto& source = a.sources[n];
    if (source.pos == 0 || !(a.used >> n & 1u)) {
      continue;
    }
    ++loaded;
    shared += std::ranges::any_of(b.sources, [&](const MatrixSources& other) { return other.pos == source.pos; });
  }
}

// Whether draw a loaded most of the position matrices it uses from where b did.
bool shares_sources(const DrawRecord& a, const DrawRecord& b) {
  uint32_t loaded = 0;
  uint32_t shared = 0;
  count_sources(a, b, loaded, shared);
  return shared * 2 > loaded;
}

// Whether matrix a (or moved, a seen through the camera move) then b can be
// one object over one frame: turned less than about 45 degrees and moved less
// than half its distance from the eye.
bool close_move(const float* a, const float* b) {
  for (int r = 0; r < 3; ++r) {
    const float* ra = a + r * 4;
    const float* rb = b + r * 4;
    const float dot = ra[0] * rb[0] + ra[1] * rb[1] + ra[2] * rb[2];
    const float na = ra[0] * ra[0] + ra[1] * ra[1] + ra[2] * ra[2];
    const float nb = rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2];
    if (!(dot > 0.7f * std::sqrt(na * nb))) {
      return false;
    }
  }
  const Vec3 tb = translation(b);
  return !(distance2(translation(a), tb) > 0.25f * length2(tb) + 1e-4f);
}

float rotation_distance2(const float* a, const float* b) {
  float sum = 0.f;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const float d = a[r * 4 + c] - b[r * 4 + c];
      sum += d * d;
    }
  }
  return sum;
}

// Matrix a seen through the camera move in the parts the flags say.
Affine through_camera(const float* a, bool rotation, Follow translate) {
  Affine out;
  std::memcpy(out.data(), a, sizeof(out));
  if (rotation) {
    Affine turn = g_camera;
    turn[3] = turn[7] = turn[11] = 0.f;
    const Affine turned = compose(turn.data(), a);
    for (int r = 0; r < 3; ++r) {
      std::memcpy(out.data() + r * 4, turned.data() + r * 4, 3 * sizeof(float));
    }
  }
  if (translate != Follow::Eye) {
    Affine turn = g_camera;
    turn[3] = turn[7] = turn[11] = 0.f;
    const Vec3 t = transform_point(translate == Follow::World ? g_camera : turn, translation(a));
    out[3] = t[0];
    out[7] = t[1];
    out[11] = t[2];
  }
  return out;
}

// Carries the parts of blended matrix m the flags say back along the camera
// move: back, or backRotation for the rotation alone.
void carry_back(float* m, bool rotation, Follow translate, const Affine& back, const Affine& backRotation) {
  const Vec3 t = translation(m);
  if (rotation) {
    const Affine turned = compose(backRotation.data(), m);
    for (int r = 0; r < 3; ++r) {
      std::memcpy(m + r * 4, turned.data() + r * 4, 3 * sizeof(float));
    }
  }
  if (translate != Follow::Eye) {
    const Vec3 moved = transform_point(translate == Follow::World ? back : backRotation, t);
    m[3] = moved[0];
    m[7] = moved[1];
    m[11] = moved[2];
  }
}

// Most objects stand still in the world, but some follow the camera (sky,
// effects around the player): either is plausible.
bool plausible_move(const float* a, const float* b) {
  return close_move(compose(g_camera.data(), a).data(), b) || close_move(a, b);
}

// Light i of a block at lights: its position, then its direction.
constexpr uint32_t light_start(uint32_t lights, uint32_t i) { return lights + i * LightFloats; }
constexpr uint32_t LightDir = 4;

double distance2d(const float* a, const float* b) {
  double sum = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    sum += d * d;
  }
  return sum;
}

double length2d(const float* a) {
  return static_cast<double>(a[0]) * a[0] + static_cast<double>(a[1]) * a[1] + static_cast<double>(a[2]) * a[2];
}

Affine turn_of(const Affine& m) {
  Affine out = m;
  out[3] = out[7] = out[11] = 0.f;
  return out;
}

// A light's position and direction seen through the camera move.
void light_through_camera(const float* in, float* out) {
  const Vec3 pos = transform_point(g_camera, {in[0], in[1], in[2]});
  const Vec3 dir = transform_point(turn_of(g_camera), {in[LightDir], in[LightDir + 1], in[LightDir + 2]});
  std::memcpy(out, pos.data(), sizeof(pos));
  std::memcpy(out + LightDir, dir.data(), sizeof(dir));
}

// GX lights and post matrices are in eye space; a light (a projected
// texture's projector) standing still in the world turns with the camera.
// Picks those of block a then b that differ, and whether each is better
// explained standing still in the world or following the camera.
void select_extras(Blend& item, const float* a, const float* b) {
  if (item.lights != 0) {
    for (uint32_t i = 0; i < ::GX::MaxLights; ++i) {
      const float* la = a + light_start(item.lights, i);
      const float* lb = b + light_start(item.lights, i);
      if (std::memcmp(la, lb, (LightDir + 3) * 4) == 0) {
        continue;
      }
      item.lightBlend |= 1u << i;
      if (!g_cameraMoved) {
        continue;
      }
      float moved[LightDir + 3];
      light_through_camera(la, moved);
      // Relative to the light's distance, which for a directional light is
      // huge: its direction alone matters then.
      const double scale = std::max(length2d(lb), 1e-8);
      const double world = distance2d(moved, lb) / scale + distance2d(moved + LightDir, lb + LightDir);
      const double eye = distance2d(la, lb) / scale + distance2d(la + LightDir, lb + LightDir);
      if (world < eye) {
        item.lightWorld |= 1u << i;
      }
    }
  }
  if (item.post != 0) {
    for (uint32_t i = 0; i < gx::MaxPTTexMtx; ++i) {
      const float* pa = a + item.post + i * MtxFloats;
      const float* pb = b + item.post + i * MtxFloats;
      if (std::memcmp(pa, pb, MtxFloats * 4) == 0) {
        continue;
      }
      bool world = false;
      Affine moved;
      if (g_cameraMoved) {
        moved = compose(pa, g_cameraInverse.data());
        float dw = 0.f;
        float de = 0.f;
        for (uint32_t q = 0; q < MtxFloats; ++q) {
          dw += (moved[q] - pb[q]) * (moved[q] - pb[q]);
          de += (pa[q] - pb[q]) * (pa[q] - pb[q]);
        }
        world = dw < de;
      }
      if (!wraps(world ? moved.data() : pa, pb)) {
        item.postBlend |= 1u << i;
        item.postWorld |= static_cast<uint32_t>(world) << i;
      }
    }
  }
}

// The older frame's draw for this one among those with its key, for draws
// whose matrices have no address to tell them by: the one exactly where it
// is, else the plausible one closest to it, since which instances of a
// model the game draws, and in what order, changes as the camera moves.
// Only one close by, or one that loaded most of its matrices from where
// this one did (a batch of instances, its current matrix unused): an
// instance just come into view would otherwise pair with another (a rock
// with the next rock along) and slide across to it. With exact, only the
// one exactly where it is.
uint32_t pick(SlotIndex& index, const DrawRecord& draw, HashType key, const float* prevBase, const float* mtx,
              bool exact) {
  const auto it = index.ranges.find(key);
  if (it == index.ranges.end()) {
    return NoMatch;
  }
  auto& range = it->second;
  while (range.next < range.end && g_used[index.slots[range.next].index]) {
    ++range.next;
  }
  // First the instance that is exactly where this one is, standing still in
  // the world or following the camera: with many instances of a model
  // (trees), the order the game draws them in changes as the camera turns.
  const Vec3 target = translation(mtx);
  const float tight = 1e-4f * length2(target) + 1e-4f;
  uint32_t found = NoMatch;
  float exactDistance = tight;
  for (uint32_t k = range.next; k < range.end; ++k) {
    const uint32_t j = index.slots[k].index;
    if (g_used[j]) {
      continue;
    }
    const Vec3 other = translation(current_matrix(prevBase, g_prev.draws[j]));
    const float distance = std::min(distance2(transform_point(g_camera, other), target), distance2(other, target));
    if (distance < exactDistance) {
      exactDistance = distance;
      found = j;
    }
  }
  if (found != NoMatch || exact) {
    return found;
  }
  const float nearby = near_enough(target);
  uint32_t best = NoMatch;
  float bestDistance = INFINITY;
  for (uint32_t k = range.next; k < range.end; ++k) {
    const uint32_t j = index.slots[k].index;
    if (g_used[j]) {
      continue;
    }
    const float* other = current_matrix(prevBase, g_prev.draws[j]);
    const float distance = distance2(transform_point(g_camera, translation(other)), target);
    if (distance < bestDistance && (distance < nearby || shares_sources(draw, g_prev.draws[j])) &&
        plausible_move(other, mtx)) {
      bestDistance = distance;
      best = j;
    }
  }
  return best;
}

constexpr uint32_t PnMask = ((1u << gx::MaxPnMtx) - 1) << 1 | ((1u << gx::MaxPnMtx) - 1) << FirstNrm;

// Copies the older block for a blend into g_older, its matrices in their
// newer slots and, in camera space, seen through the camera move.
void store_older(Blend& item, const float* a) {
  item.prevOffset = static_cast<uint32_t>(g_older.size());
  g_older.resize(item.prevOffset + item.extent);
  float* out = g_older.data() + item.prevOffset;
  std::memcpy(out, a, item.extent * 4);
  for (uint32_t lights = item.lightWorld; lights != 0; lights &= lights - 1) {
    const uint32_t start = light_start(item.lights, std::countr_zero(lights));
    light_through_camera(a + start, out + start);
  }
  for (uint32_t post = item.postWorld; post != 0; post &= post - 1) {
    const uint32_t start = item.post + std::countr_zero(post) * MtxFloats;
    const Affine moved = compose(a + start, g_cameraInverse.data());
    std::memcpy(out + start, moved.data(), sizeof(moved));
  }
  for (uint32_t matrices = item.matrices & PnMask; matrices != 0; matrices &= matrices - 1) {
    const uint32_t n = std::countr_zero(matrices);
    const Affine moved =
        through_camera(a + prev_start(item, n), item.cameraRotation, n < FirstNrm ? item.cameraTranslation : Follow::Eye);
    std::memcpy(out + matrix_start(n), moved.data(), sizeof(moved));
  }
}

// Whether matrix m's rotation is a plain scale: in eye space, only what is
// made to face the camera is.
bool faces_eye(const float* m) {
  const float scale = std::sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
  if (!(scale > 1e-12f)) {
    return false;
  }
  const float tolerance = 1e-3f * scale;
  return std::abs(m[1]) < tolerance && std::abs(m[2]) < tolerance && std::abs(m[4]) < tolerance &&
         std::abs(m[6]) < tolerance && std::abs(m[8]) < tolerance && std::abs(m[9]) < tolerance;
}

// How the matched draws of a model last followed the camera, for its
// instances that come into view; kept across frames, as a whole group of
// them (a stretch of trees) can come into view at once.
struct Modes {
  bool rotation;
  Follow translation;
};
absl::flat_hash_map<HashType, Modes> g_keyModes;
absl::flat_hash_map<HashType, Modes> g_groupModes;
std::vector<uint32_t> g_unmatched;

// A draw the older frame lacks (it just came into view, as the camera turns
// toward it) is taken to stand still in the world, or to follow the camera
// as the model's other instances do (billboards face it): in the newer eye
// space its older matrices are its newer ones. Not for screen-space draws,
// whose projection is orthographic.
void reproject(const DrawRecord& draw, const float* b) {
  const uint16_t camera = camera_of(draw);
  use_camera(camera);
  if (!g_cameraMoved || b[15] != 0.f || !g_blocks.insert(draw.offset).second) {
    return;
  }
  Modes modes{true, Follow::World};
  if (camera != 0) {
    // In the world, seen through its camera.
  } else if (const auto it = g_keyModes.find(draw.key); it != g_keyModes.end()) {
    modes = it->second;
  } else if (const auto group = g_groupModes.find(draw.group); group != g_groupModes.end()) {
    modes = group->second;
  } else if (faces_eye(b + matrix_start(1 + std::min(draw.current, gx::MaxPnMtx - 1)))) {
    // Unseen yet, and unturned in eye space: a billboard.
    modes.rotation = false;
  }
  Blend item{draw.offset / 4, 0, PnMask, {}, modes.rotation, modes.translation};
  item.camera = camera;
  item.prevOffset = static_cast<uint32_t>(g_older.size());
  g_older.insert(g_older.end(), b, b + BlendFloats);
  g_blends.push_back(item);
  g_dirtyBegin = std::min(g_dirtyBegin, draw.offset);
  g_dirtyEnd = std::max(g_dirtyEnd, draw.offset + BlendBytes);
}

// The older frame's draw with this one's identity. Some draws share one: a
// model's instances whose matrices the game builds in one scratch buffer
// (distant trees). Among those, as in pick, the one exactly where this one
// is, else the plausible one closest to it. With exact, only the one exactly
// where it is, or the one draw with the identity.
uint32_t pick_identity(const IdentityIndex& index, const DrawRecord& draw, const float* prevBase, const float* mtx,
                       bool exact) {
  const uint64_t id = identity(draw, index.hash);
  const auto it = id == 0 ? index.first.end() : index.first.find(id);
  if (it == index.first.end()) {
    return NoMatch;
  }
  uint32_t j = it->second;
  if (index.next[j] == NoMatch) {
    return g_used[j] ? NoMatch : j;
  }
  while (j != NoMatch && g_used[j]) {
    j = index.next[j];
  }
  if (j == NoMatch || (!exact && index.next[j] == NoMatch)) {
    return j;
  }
  // A skinned model's instances can share the matrix placing them (a common
  // root), and so stand at the same place: the one whose bones were loaded
  // from where this one's were, else one instance's skeleton blends into
  // another's.
  uint32_t mostShared = 0;
  uint32_t sharing = NoMatch;
  uint32_t loaded = 0;
  for (uint32_t k = j; k != NoMatch; k = index.next[k]) {
    if (g_used[k]) {
      continue;
    }
    uint32_t shared = 0;
    loaded = 0; // the same for every candidate
    count_sources(draw, g_prev.draws[k], loaded, shared);
    if (shared > mostShared) {
      mostShared = shared;
      sharing = k;
    }
  }
  if (loaded >= 2 && mostShared * 2 > loaded) {
    return sharing;
  }
  const Vec3 target = translation(mtx);
  const float tight = 1e-4f * length2(target) + 1e-4f;
  const float nearby = near_enough(target);
  uint32_t best = NoMatch;
  float bestDistance = INFINITY;
  for (; j != NoMatch; j = index.next[j]) {
    if (g_used[j]) {
      continue;
    }
    const float* other = current_matrix(prevBase, g_prev.draws[j]);
    const Vec3 at = translation(other);
    const float distance = std::min(distance2(transform_point(g_camera, at), target), distance2(at, target));
    if (distance < tight) {
      return j;
    }
    if (exact) {
      continue;
    }
    if (distance < bestDistance && (distance < nearby || shares_sources(draw, g_prev.draws[j])) &&
        plausible_move(other, mtx)) {
      bestDistance = distance;
      best = j;
    }
  }
  return best;
}

// Whether matrix a then b stands exactly still in the world.
bool stands_still(const float* a, const float* b) {
  const Affine moved = compose(g_camera.data(), a);
  const Vec3 tb = translation(b);
  const float scale = b[0] * b[0] + b[4] * b[4] + b[8] * b[8];
  return distance2(translation(moved.data()), tb) < 1e-6f * length2(tb) + 1e-6f &&
         rotation_distance2(moved.data(), b) < 1e-6f * scale;
}

// Pairs each draw of the newer frame with a draw of the older one and keeps
// the pairs worth blending.
void match() {
  g_blends.clear();
  g_dirtyBegin = UINT32_MAX;
  g_dirtyEnd = 0;
  const auto& prev = g_prev.draws;
  if (prev.empty() || g_cur.draws.empty()) {
    return;
  }
  g_blocks.clear();
  g_older.clear();
  if (g_keyModes.size() > 65536) {
    g_keyModes.clear();
    g_groupModes.clear();
  }
  g_unmatched.clear();
  build_index(g_byKey, prev, &DrawRecord::key);
  build_index(g_byGroup, prev, &DrawRecord::group);
  build_identities(g_byKeyIdentity, prev);
  build_identities(g_byGroupIdentity, prev);
  g_used.assign(prev.size(), 0);
  estimate_camera();
  exact_cameras();
  const auto* prevBase = reinterpret_cast<const float*>(g_prev.uniforms.data());
  const auto* curBase = reinterpret_cast<const float*>(g_cur.uniforms.data());
  const size_t prevSize = g_prev.uniforms.size();
  const size_t curSize = g_cur.uniforms.size();
  // Each draw exactly where an older one is takes it first: one just come
  // into view next to another instance of its model (a pair of trees) would
  // otherwise take that one's older draw as the closest and slide across.
  g_pairs.assign(g_cur.draws.size(), NoMatch);
  for (const bool exact : {true, false}) {
    for (size_t i = 0; i < g_cur.draws.size(); ++i) {
      const auto& draw = g_cur.draws[i];
      if (g_pairs[i] != NoMatch || draw.offset + BlendBytes > curSize) {
        continue;
      }
      use_camera(camera_of(draw));
      const float* mtx = current_matrix(curBase, draw);
      uint32_t j = pick_identity(g_byKeyIdentity, draw, prevBase, mtx, exact);
      if (j == NoMatch) {
        j = pick_identity(g_byGroupIdentity, draw, prevBase, mtx, exact);
      }
      if (j == NoMatch) {
        j = pick(g_byKey, draw, draw.key, prevBase, mtx, exact);
      }
      if (j == NoMatch) {
        j = pick(g_byGroup, draw, draw.group, prevBase, mtx, exact);
      }
      if (j != NoMatch) {
        g_used[j] = 1;
        g_pairs[i] = j;
      }
    }
  }
  uint32_t matched = 0;
  uint32_t implausible = 0;
  for (size_t i = 0; i < g_cur.draws.size(); ++i) {
    const auto& draw = g_cur.draws[i];
    if (draw.offset + BlendBytes > curSize) {
      continue;
    }
    const uint32_t j = g_pairs[i];
    if (j == NoMatch) {
      g_unmatched.push_back(static_cast<uint32_t>(i));
      continue;
    }
    const auto& other = prev[j];
    const uint16_t camera = camera_of(draw);
    use_camera(camera);
    // Draws with the same GX state share a uniform block; the first match
    // blends it.
    if (draw.offset + BlendBytes > curSize || other.offset + BlendBytes > prevSize ||
        draw.current >= gx::MaxPnMtx || !g_blocks.insert(draw.offset).second) {
      continue;
    }
    const float* a = prevBase + other.offset / 4;
    const float* b = curBase + draw.offset / 4;
    Blend item{draw.offset / 4, other.offset / 4, 0, {}};
    item.camera = camera;
    if (other.lights == draw.lights && other.post == draw.post) {
      const uint32_t extent = std::max({BlendFloats, draw.lights != 0 ? draw.lights + LightsFloats : 0u,
                                        draw.post != 0 ? draw.post + PostFloats : 0u});
      if (draw.offset + extent * 4 <= curSize && other.offset + extent * 4 <= prevSize) {
        item.lights = draw.lights;
        item.post = draw.post;
        item.extent = extent;
      }
    }
    if (std::memcmp(a, b, item.extent * 4) == 0) {
      continue;
    }
    for (uint32_t i = 0; i < gx::MaxPnMtx; ++i) {
      item.prevSlot[i] = pair_slot(draw.sources[i].pos, i, other, &MatrixSources::pos);
      item.prevSlot[gx::MaxPnMtx + i] = pair_slot(draw.sources[i].nrm, i, other, &MatrixSources::nrm);
      // The game builds some models' matrices in a scratch buffer whose order
      // changes from frame to frame: the same slot of a draw paired by where
      // it is, if it stands exactly still there. (Not a moving one, whose
      // slot may then hold another bone.)
      if (item.prevSlot[i] == NoSlot && draw.sources[i].pos != 0 && other.sources[i].pos != 0 &&
          stands_still(a + matrix_start(1 + i), b + matrix_start(1 + i))) {
        item.prevSlot[i] = static_cast<uint8_t>(i);
        if (item.prevSlot[gx::MaxPnMtx + i] == NoSlot && other.sources[i].nrm != 0) {
          item.prevSlot[gx::MaxPnMtx + i] = static_cast<uint8_t>(i);
        }
      }
    }
    if (item.prevSlot[draw.current] == NoSlot) {
      continue;
    }
    ++matched;
    // An object standing still in the world is blended in the newer eye
    // space and carried back along the camera move; one that follows the
    // camera (sky, effects around the player), as it is.
    const float* currentA = a + prev_start(item, 1 + draw.current);
    const float* currentB = b + matrix_start(1 + draw.current);
    if (camera != 0) {
      // Built with a camera the game gave: wherever the object goes, it goes
      // there in the world.
      item.cameraRotation = g_cameraMoved;
      item.cameraTranslation = g_cameraMoved ? Follow::World : Follow::Eye;
    } else if (g_cameraMoved) {
      // Whichever of the two explains the move better, for its rotation and
      // its translation each.
      const Affine moved = compose(g_camera.data(), currentA);
      item.cameraRotation = rotation_distance2(moved.data(), currentB) < rotation_distance2(currentA, currentB);
      Affine turn = g_camera;
      turn[3] = turn[7] = turn[11] = 0.f;
      const Vec3 target = translation(currentB);
      const float eye = distance2(translation(currentA), target);
      const float world = distance2(translation(moved.data()), target);
      const float turned = distance2(transform_point(turn, translation(currentA)), target);
      item.cameraTranslation = world <= std::min(eye, turned) ? Follow::World
                               : turned < eye                  ? Follow::Turn
                                                               : Follow::Eye;
    }
    const bool cameraSpace = item.cameraRotation || item.cameraTranslation != Follow::Eye;
    if (!close_move(through_camera(currentA, item.cameraRotation, item.cameraTranslation).data(), currentB)) {
      ++implausible;
      continue;
    }
    for (uint32_t n = 0; n < MatrixCount; ++n) {
      const bool pn = n != 0 && !is_texture_matrix(n);
      if (pn ? item.prevSlot[n < FirstNrm ? n - 1 : gx::MaxPnMtx + n - FirstNrm] == NoSlot : n != 0 && wraps(a + matrix_start(n), b + matrix_start(n))) {
        continue;
      }
      const uint32_t start = matrix_start(n);
      if ((pn && cameraSpace) || std::memcmp(a + prev_start(item, n), b + start, matrix_floats(n) * 4) != 0) {
        item.matrices |= 1u << n;
      }
    }
    // Even paired by source, a position matrix (and its normal matrix) is
    // blended only where the pair plausibly is one bone.
    for (uint32_t i = 0; i < gx::MaxPnMtx; ++i) {
      const uint32_t pos = 1 + i;
      if (!(item.matrices >> pos & 1u)) {
        continue;
      }
      const float* pa = a + prev_start(item, pos);
      const float* pb = b + matrix_start(pos);
      if (!close_move(through_camera(pa, item.cameraRotation, item.cameraTranslation).data(), pb)) {
        item.matrices &= ~((1u << pos) | (1u << (FirstNrm + i)));
      }
    }
    select_extras(item, a, b);
    if (item.matrices == 0 && item.lightBlend == 0 && item.postBlend == 0) {
      continue;
    }
    store_older(item, a);
    if (g_cameraMoved && camera == 0) {
      const Modes modes{item.cameraRotation, item.cameraTranslation};
      g_keyModes.insert_or_assign(draw.key, modes);
      g_groupModes.insert_or_assign(draw.group, modes);
    }
    g_blends.push_back(item);
    g_dirtyBegin = std::min(g_dirtyBegin, draw.offset);
    g_dirtyEnd = std::max(g_dirtyEnd, draw.offset + item.extent * 4);
  }
  for (const uint32_t i : g_unmatched) {
    const auto& draw = g_cur.draws[i];
    reproject(draw, curBase + draw.offset / 4);
  }
  if (implausible * 2 > matched) {
    // Most of the scene jumped: a camera cut. Show the new frame as it is.
    g_blends.clear();
  }
}

// A lerped rotation shrinks toward the middle of the turn; give each column
// of the 3x3 part its lerped length back, which keeps the object's scale.
inline void renormalize(float* m, const float* a, const float* b, float t) {
  for (int c = 0; c < 3; ++c) {
    const auto length = [c](const float* x) {
      return std::sqrt(x[c] * x[c] + x[4 + c] * x[4 + c] + x[8 + c] * x[8 + c]);
    };
    const float have = length(m);
    if (have > 1e-12f) {
      const float scale = (length(a) + (length(b) - length(a)) * t) / have;
      m[c] *= scale;
      m[4 + c] *= scale;
      m[8 + c] *= scale;
    }
  }
}

struct WayBack {
  Affine back;
  Affine rotation;
  Affine inverse;
};
std::vector<WayBack> g_backs;

// Writes the blended uniform stream at t into g_buffer.
void blend(float t) {
  auto* dst = reinterpret_cast<float*>(g_scratch.data());
  const auto* curBase = reinterpret_cast<const float*>(g_cur.uniforms.data());
  // For each camera move, the way back from the newer eye space to the one at
  // t: as a whole, its rotation alone, and its inverse.
  g_backs.resize(g_moves.size());
  for (size_t k = 0; k < g_moves.size(); ++k) {
    const auto& move = g_moves[k];
    auto& way = g_backs[k];
    way.back = Identity;
    way.inverse = Identity;
    if (move.moved) {
      way.back = screw_at(move.screw, 1.f - t);
      for (int q = 0; q < 12; ++q) {
        way.back[q] += move.screwError[q] * (1.f - t);
      }
      if (!invert(way.back.data(), way.inverse)) {
        way.inverse = Identity;
      }
    }
    way.rotation = turn_of(way.back);
  }
  for (const auto& item : g_blends) {
    const auto& [back, backRotation, backInverse] = g_backs[item.camera];
    float* out = dst + item.offset;
    const float* a = g_older.data() + item.prevOffset;
    const float* b = curBase + item.offset;
    for (uint32_t matrices = item.matrices; matrices != 0; matrices &= matrices - 1) {
      const uint32_t n = std::countr_zero(matrices);
      const uint32_t start = matrix_start(n);
      const uint32_t end = start + matrix_floats(n);
      if (t >= 1.f) {
        std::memcpy(out + start, b + start, (end - start) * 4);
        continue;
      }
      for (uint32_t i = start; i < end; ++i) {
        out[i] = a[i] + (b[i] - a[i]) * t;
      }
      if (n != 0 && !is_texture_matrix(n)) {
        renormalize(out + start, a + start, b + start, t);
        carry_back(out + start, item.cameraRotation, n < FirstNrm ? item.cameraTranslation : Follow::Eye, back, backRotation);
      }
    }
    for (uint32_t lights = item.lightBlend; lights != 0; lights &= lights - 1) {
      const uint32_t i = std::countr_zero(lights);
      const uint32_t start = light_start(item.lights, i);
      if (t >= 1.f) {
        std::memcpy(out + start, b + start, (LightDir + 3) * 4);
        continue;
      }
      const float* la = a + start;
      const float* lb = b + start;
      Vec3 pos;
      Vec3 dir;
      for (int c = 0; c < 3; ++c) {
        pos[c] = la[c] + (lb[c] - la[c]) * t;
        dir[c] = la[LightDir + c] + (lb[LightDir + c] - la[LightDir + c]) * t;
      }
      // A lerped direction shrinks toward the middle of a turn.
      const float have = std::sqrt(length2(dir));
      if (have > 1e-12f) {
        const float na = static_cast<float>(std::sqrt(length2d(la + LightDir)));
        const float nb = static_cast<float>(std::sqrt(length2d(lb + LightDir)));
        const float scale = (na + (nb - na) * t) / have;
        dir = {dir[0] * scale, dir[1] * scale, dir[2] * scale};
      }
      if (item.lightWorld >> i & 1u) {
        pos = transform_point(back, pos);
        dir = transform_point(backRotation, dir);
      }
      std::memcpy(out + start, pos.data(), sizeof(pos));
      std::memcpy(out + start + LightDir, dir.data(), sizeof(dir));
    }
    for (uint32_t post = item.postBlend; post != 0; post &= post - 1) {
      const uint32_t i = std::countr_zero(post);
      const uint32_t start = item.post + i * MtxFloats;
      if (t >= 1.f) {
        std::memcpy(out + start, b + start, MtxFloats * 4);
        continue;
      }
      Affine m;
      for (uint32_t q = 0; q < MtxFloats; ++q) {
        m[q] = a[start + q] + (b[start + q] - a[start + q]) * t;
      }
      // It takes eye space at t: through the way forward to the newer one.
      if (item.postWorld >> i & 1u) {
        m = compose(m.data(), backInverse.data());
      }
      std::memcpy(out + start, m.data(), sizeof(m));
    }
  }
  if (!g_uploaded) {
    // The first replay of a frame writes all of it; later ones only the span
    // the blended blocks cover, the rest is unchanged.
    g_queue.WriteBuffer(g_buffer, 0, g_scratch.data(), AURORA_ALIGN(g_scratch.size(), 4));
    g_uploaded = true;
  } else if (g_dirtyEnd > g_dirtyBegin) {
    g_queue.WriteBuffer(g_buffer, g_dirtyBegin, g_scratch.data() + g_dirtyBegin, g_dirtyEnd - g_dirtyBegin);
  }
}

// How long before it is needed a frame should arrive: what its arrival
// jitters by.
constexpr Clock::duration Slack = std::chrono::milliseconds{4};

// Submits the held frame and starts interpolating toward it from base.
void activate(Clock::time_point base) {
  g_next.submit();
  g_next.submit = {};
  g_prev = std::move(g_cur);
  g_prev.replay = {};
  g_prev.present = {};
  g_cur = std::move(g_next);
  g_next = {};
  g_base = base;
  match();
  ++g_stats.interpFrames;
  g_stats.interpDraws += static_cast<uint32_t>(g_cur.draws.size());
  g_stats.interpBlendedDraws += static_cast<uint32_t>(g_blends.size());
  g_scratch.clear();
  g_scratch.append(g_cur.uniforms.data(), g_cur.uniforms.size());
  g_uploaded = false;
}

// Debug (AURORA_INTERP_CHECK=<dir>): renders the shown frame at t into a
// PPM there. Interpolation is right where the newer frame at t = 0 matches
// the older one at t = 1.
void capture(float t, const char* name) {
  static const char* dir = std::getenv("AURORA_INTERP_CHECK");
  const auto& source = webgpu::present_source();
  const uint32_t width = source.size.width;
  const uint32_t height = source.size.height;
  const uint32_t row = AURORA_ALIGN(width * 4, 256);
  const wgpu::BufferDescriptor bufferDescriptor{
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = uint64_t{row} * height,
  };
  auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  auto encoder = g_device.CreateCommandEncoder();
  blend(t);
  g_vertexUniformBindGroup = &g_bindGroup;
  g_replaying.store(true, std::memory_order_relaxed);
  g_cur.replay(encoder);
  g_replaying.store(false, std::memory_order_relaxed);
  g_vertexUniformBindGroup = &g_uniformBindGroup;
  const wgpu::TexelCopyTextureInfo src{.texture = source.texture};
  const wgpu::TexelCopyBufferInfo dst{.layout = {.bytesPerRow = row, .rowsPerImage = height}, .buffer = buffer};
  const wgpu::Extent3D extent{width, height, 1};
  encoder.CopyTextureToBuffer(&src, &dst, &extent);
  const auto commands = encoder.Finish();
  g_queue.Submit(1, &commands);
  bool done = false;
  const auto future = buffer.MapAsync(wgpu::MapMode::Read, 0, bufferDescriptor.size, wgpu::CallbackMode::WaitAnyOnly,
                                      [&](wgpu::MapAsyncStatus, wgpu::StringView) { done = true; });
  webgpu::g_instance.WaitAny(future, UINT64_MAX);
  if (!done) {
    return;
  }
  const auto* pixels = static_cast<const uint8_t*>(buffer.GetConstMappedRange());
  const bool bgra = source.format == wgpu::TextureFormat::BGRA8Unorm || source.format == wgpu::TextureFormat::BGRA8UnormSrgb;
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s.ppm", dir, name);
  if (FILE* file = std::fopen(path, "wb")) {
    std::fprintf(file, "P6 %u %u 255%c", width, height, 10);
    std::vector<uint8_t> line(width * 3);
    for (uint32_t y = 0; y < height; ++y) {
      const uint8_t* in = pixels + size_t{row} * y;
      for (uint32_t x = 0; x < width; ++x) {
        line[x * 3 + 0] = in[x * 4 + (bgra ? 2 : 0)];
        line[x * 3 + 1] = in[x * 4 + 1];
        line[x * 3 + 2] = in[x * 4 + (bgra ? 0 : 2)];
      }
      std::fwrite(line.data(), 1, line.size(), file);
    }
    std::fclose(file);
  }
  buffer.Unmap();
}

float time_at(Clock::time_point shown) {
  return std::chrono::duration<float>(shown - g_base).count() / std::chrono::duration<float>(g_period).count();
}

// Render worker: replays the newest frame at the current time and presents it.
void present_tick() {
  if (!g_cur.replay) {
    return;
  }
  const auto start = Clock::now();
  // When this replay will be on screen. Presents go out a refresh apart, so
  // that is one interval after the last one's, pulled slowly toward what the
  // clock says: the time a tick starts at jitters with the work before it.
  const auto measured = start + g_interval;
  const auto predicted = g_shown + g_interval;
  const auto drift = measured - predicted;
  const auto shown = drift < g_interval / 2 && drift > -g_interval / 2 ? predicted + drift / 16 : measured;
  g_shown = shown;
  if (time_at(shown) >= 1.f && g_next.submit) {
    // On to the held frame, keeping the timeline continuous; the base drifts
    // toward the frame arriving Slack before it is needed.
    const auto waited = start - g_next.arrival;
    static const bool check = std::getenv("AURORA_INTERP_CHECK") != nullptr;
    static unsigned checkFrame = 0;
    const bool capturing = check && ++checkFrame % 15 == 0 && checkFrame / 15 <= 40;
    char name[64];
    if (capturing) {
      std::snprintf(name, sizeof(name), "%03u_a_old_t1", checkFrame / 15);
      capture(1.f, name);
    }
    activate(g_base + g_period - (waited - Slack) / 16);
    if (capturing) {
      std::snprintf(name, sizeof(name), "%03u_b_new_t0", checkFrame / 15);
      capture(0.f, name);
      std::snprintf(name, sizeof(name), "%03u_c_new_t1", checkFrame / 15);
      capture(1.f, name);
    }
  }
  const float unclamped = time_at(shown);
  const float t = std::clamp(unclamped, 0.f, 1.f);
  const bool last = (t >= 1.f || g_blends.empty()) && !g_next.submit;
  // Past the end with nothing newer while the game is still sending frames:
  // the picture stops until the late one arrives. A paused game's last frame
  // runs far past the end and isn't counted.
  if (unclamped > 1.f && unclamped < 3.f && !g_next.submit) {
    ++g_stats.interpStarved;
  }
  static constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "Interpolated frame encoder"};
  auto encoder = g_device.CreateCommandEncoder(&EncoderDescriptor);
  // Every replay of a frame binds the blended copy, so its render bundles
  // are recorded once.
  static const bool noBlend = std::getenv("AURORA_INTERP_NOBLEND") != nullptr;
  blend(last || noBlend ? 1.f : t);
  g_vertexUniformBindGroup = &g_bindGroup;
  g_replaying.store(true, std::memory_order_relaxed);
  g_cur.replay(encoder);
  g_replaying.store(false, std::memory_order_relaxed);
  g_vertexUniformBindGroup = &g_uniformBindGroup;
  g_stats.interpMicros +=
      static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
  g_cur.present(encoder);
  ++g_stats.interpPresents;
  const auto presented = Clock::now();
  const auto sinceLast = presented - g_lastPresent;
  if (sinceLast > g_interval * 3 / 2 && sinceLast < std::chrono::milliseconds{250}) {
    ++g_stats.interpLatePresents;
  }
  g_lastPresent = presented;
  if (last && t >= 1.f) {
    std::lock_guard lock{g_mutex};
    g_pending = false;
  }
}

void sleep_until(Clock::time_point target, [[maybe_unused]] void* timer) {
  const auto now = Clock::now();
  if (target <= now) {
    return;
  }
#ifdef _WIN32
  LARGE_INTEGER due;
  due.QuadPart = -std::chrono::duration_cast<std::chrono::nanoseconds>(target - now).count() / 100;
  if (timer != nullptr && SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) {
    WaitForSingleObject(timer, INFINITE);
    return;
  }
#endif
  std::this_thread::sleep_until(target);
}

// Asks the render worker for one replay at a time: back to back with vsync,
// which paces them, or every g_interval without.
void pacer_main() {
#ifdef _WIN32
  HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
#else
  void* timer = nullptr;
#endif
  auto next = Clock::now();
  std::unique_lock lock{g_mutex};
  while (true) {
    g_cv.wait(lock, [] { return g_stop || g_pending; });
    if (g_stop) {
      break;
    }
    if (!g_vsync) {
      lock.unlock();
      sleep_until(next, timer);
      const auto now = Clock::now();
      next = std::max(next + g_interval, now);
      lock.lock();
      if (g_stop) {
        break;
      }
    }
    g_ticking = true;
    lock.unlock();
    render_worker::enqueue_work([] {
      present_tick();
      {
        std::lock_guard tickLock{g_mutex};
        g_ticking = false;
      }
      g_cv.notify_all();
    });
    lock.lock();
    g_cv.wait(lock, [] { return g_stop || !g_ticking; });
  }
#ifdef _WIN32
  if (timer != nullptr) {
    CloseHandle(timer);
  }
#endif
}
} // namespace

void initialize(bool vsync, uint32_t presentRate) {
  g_vsync = vsync;
  float rate = static_cast<float>(presentRate);
  if (rate <= 0.f) {
    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window::get_sdl_window()));
    rate = mode != nullptr && mode->refresh_rate > 0.f ? mode->refresh_rate : 60.f;
  }
  g_interval = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<float>(1.f / rate));

  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Interpolated uniform buffer",
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = UniformBufferSize,
  };
  g_buffer = g_device.CreateBuffer(&bufferDescriptor);
  const std::array entries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .buffer = g_buffer,
          .size = gx::MaxUniformSize,
      },
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .label = "Interpolated uniform bind group",
      .layout = g_uniformBindGroupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  g_stop = false;
  g_pending = false;
  g_ticking = false;
  g_enabled = true;
  g_pacer = std::thread(pacer_main);
  Log.info("Frame interpolation on, presenting {} at {:.0f} Hz", vsync ? "with vsync" : "paced", rate);
}

void shutdown() {
  if (!g_enabled) {
    return;
  }
  {
    std::lock_guard lock{g_mutex};
    g_stop = true;
  }
  g_cv.notify_all();
  if (g_pacer.joinable()) {
    g_pacer.join();
  }
  render_worker::synchronize();
  if (g_next.submit) {
    g_next.submit();
  }
  g_next = {};
  g_enabled = false;
  g_prev = {};
  g_cur = {};
  g_scratch.release();
  g_blends.clear();
  g_bindGroup = {};
  g_buffer = {};
  g_vertexUniformBindGroup = &g_uniformBindGroup;
}

void discard() {
  if (!g_enabled) {
    return;
  }
  const auto drop = [] {
    if (g_next.submit) {
      g_next.submit();
    }
    g_next = {};
    g_prev = {};
    g_cur = {};
    g_blends.clear();
  };
  if (render_worker::is_worker_thread()) {
    // The pacer waits for this tick of the worker's to end before its next.
    std::lock_guard lock{g_mutex};
    g_pending = false;
    drop();
    return;
  }
  {
    std::unique_lock lock{g_mutex};
    g_pending = false;
    g_cv.wait(lock, [] { return !g_ticking; });
  }
  render_worker::enqueue_work(drop);
  render_worker::synchronize();
}

bool enabled() noexcept { return g_enabled; }

bool replaying() noexcept { return g_replaying.load(std::memory_order_relaxed); }

void push_frame(ByteBuffer uniforms, std::vector<DrawRecord> draws, std::vector<ViewMatrix> cameras, ReplayFn replay,
                EndFrameCallback present, SubmitFn submit) {
  const auto now = Clock::now();
  if (g_lastArrival != Clock::time_point{}) {
    const auto delta = now - g_lastArrival;
    if (delta < std::chrono::milliseconds{100}) {
      g_period += (delta - g_period) / 8;
    }
  }
  g_lastArrival = now;
  if (g_next.submit) {
    // The game got ahead of the replays: drop the rest of the frame shown.
    activate(g_base + g_period);
  }
  g_next = Frame{std::move(uniforms), std::move(draws), std::move(cameras), std::move(replay), std::move(present),
                 std::move(submit), now};
  std::lock_guard lock{g_mutex};
  if (!g_cur.replay || !g_pending) {
    // Nothing left to show of the frame before: start on this one, a little
    // late so the next one has time to arrive.
    activate(now + Slack);
  }
  g_pending = true;
  g_cv.notify_all();
}

} // namespace aurora::gfx::interp
