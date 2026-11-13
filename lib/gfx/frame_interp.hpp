#pragma once

// Frame interpolation. The game renders at its own rate (30 FPS for most GX
// titles); between two of its frames the render worker replays the newer one
// with every draw's matrices blended toward the matching draw of the older
// one, and presents each replay. A game frame is shown one frame late, so each
// replay sits between two frames the game really produced.
//
// Draws are matched across frames by a key (pipeline, textures, sizes) and
// their order among draws with the same key; within a draw, matrices are
// paired by where the game loaded them from, as a skinned model's bones can
// land in different matrix slots each frame. A draw with no match, or whose
// matrices moved too far to be the same object (a camera cut), is drawn as the
// newer frame has it.
//
// A game can say which camera a pass of draws was built with (see
// record_camera): those draws' objects are blended in the world and seen
// through the camera's own move. Elsewhere the camera move is estimated from
// the draws, and whether each object stands in the world or follows the
// camera is guessed.
//
// 2D layouts (menus, HUD) are often placed on the CPU: their screen-space
// draws keep one matrix and carry the positions in their vertices. Those
// vertices are blended too, written over the frame's own in the vertex buffer
// for each replay.

#include "common.hpp"

#include <array>
#include <span>
#include <vector>

namespace aurora::gfx::interp {

// What loaded a position matrix and its normal matrix: 1 + the array index
// of an indexed load, 0 for an inline one.
struct MatrixSources {
  uint32_t pos = 0;
  uint32_t nrm = 0;
};
constexpr uint32_t MatrixSlots = 10;

// Where a GX uniform block's parts sit: its matrices in the uniform stream
// (bytes), then its lights and post-transform texture matrices in floats
// from there (0 for none). Lights are in eye space, and the post matrices
// often take eye-space input (projected shadows), so both move with the
// camera.
struct BlockLayout {
  uint32_t matrices = 0;
  uint16_t lights = 0;
  uint16_t post = 0;
};

// A view matrix, row-major 3x4 like GX position matrices: the world to eye
// space.
using ViewMatrix = std::array<float, 12>;

// How a screen-space draw's vertices hold their positions, when they hold
// them directly (stride 0: they don't). type is the GXCompType (GX_F32 or
// GX_S16), frac its fixed-point bits.
struct VertexLayout {
  uint16_t stride = 0;
  uint8_t position = 0; // byte offset of the position in a vertex
  uint8_t type = 0;
  uint8_t count = 0; // 2 or 3 components
  uint8_t frac = 0;
  bool bigEndian = false;

  bool operator==(const VertexLayout&) const = default;
};

struct DrawRecord {
  HashType key;   // pipeline, textures and sizes
  HashType group; // pipeline and textures, for draws whose sizes changed
  uint32_t offset;  // the matrices of the draw's uniform block in the frame's uniform stream
  uint32_t current; // the position matrix placing it: the one vertices use without a matrix index, else the first they pick
  std::array<MatrixSources, MatrixSlots> sources;
  uint16_t lights;  // as in BlockLayout
  uint16_t post;
  uint16_t camera;  // 1 + the frame's camera the draw's matrices were built with, 0 if unknown
  uint16_t used;    // bit n: its vertices use position matrix n
  // With vertices.stride set: where its vertices are in the vertex buffer,
  // and in the frame's CPU copy of such vertices (vertBytes 0: none kept).
  uint32_t vertOffset = 0;
  VertexLayout vertices{};
  uint32_t vertData = 0;
  uint32_t vertBytes = 0;
};

// Encodes the retained frame's passes again into the encoder.
using ReplayFn = std::function<void(rhi::CommandList&)>;
// Submits a frame's own commands (its copies and side effects) and releases
// its frame slot.
using SubmitFn = std::function<void()>;

void initialize(bool vsync, uint32_t presentRate);
void shutdown();
bool enabled() noexcept;
// True while a replay is being encoded: one-shot work (readback snapshots)
// is left to the frame's own encode.
bool replaying() noexcept;

// The vertex-stage uniform bind group draws bind: the frame's own, or the
// blended copy during a replay.
extern rhi::BindGroup* g_vertexUniformBindGroup;

// Stops the replays and drops the frames held for them (submitting the
// newest), as the render targets they draw into are about to be replaced.
// Interpolation resumes with the next frame.
void discard();

// Render thread, while recording: notes a GX draw's uniform block, and keeps
// a copy of its vertices when record.vertices.stride is set.
void record_draw(const DrawRecord& record, std::span<const uint8_t> vertices);
// Render thread, while recording: a draw merged into the last one recorded
// adds the position matrices its vertices use, and its vertices.
void merge_draw(uint16_t used, std::span<const uint8_t> vertices);
// Render thread, while recording: notes a camera of the frame, returning the
// value for DrawRecord::camera.
uint16_t record_camera(const ViewMatrix& view);

// Render worker: a frame was recorded. uniforms is a CPU copy of its uniform
// stream, vertices a copy of the vertices its draws keep; submit is called
// once the frame before it has been fully shown (its replays need the GPU
// buffers as that frame left them), then it is replayed; present finishes an encoder, submits it and presents.
void push_frame(ByteBuffer uniforms, std::vector<DrawRecord> draws, std::vector<ViewMatrix> cameras,
                std::vector<uint8_t> vertices, ReplayFn replay, EndFrameCallback present, SubmitFn submit);

} // namespace aurora::gfx::interp
