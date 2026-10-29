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

#include "common.hpp"

#include <array>
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
};

// Encodes the retained frame's passes again into the encoder.
using ReplayFn = std::function<void(wgpu::CommandEncoder&)>;
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
extern const wgpu::BindGroup* g_vertexUniformBindGroup;

// Stops the replays and drops the frames held for them (submitting the
// newest), as the render targets they draw into are about to be replaced.
// Interpolation resumes with the next frame.
void discard();

// Render thread, while recording: notes a GX draw's uniform block.
void record_draw(const DrawRecord& record);
// Render thread, while recording: a draw merged into the last one recorded
// adds the position matrices its vertices use.
void merge_draw(uint16_t used);
// Render thread, while recording: notes a camera of the frame, returning the
// value for DrawRecord::camera.
uint16_t record_camera(const ViewMatrix& view);

// Render worker: a frame was recorded. uniforms is a CPU copy of its uniform
// stream; submit is called once the frame before it has been fully shown (its
// replays need the GPU buffers as that frame left them), then it is replayed;
// present finishes an encoder, submits it and presents.
void push_frame(ByteBuffer uniforms, std::vector<DrawRecord> draws, std::vector<ViewMatrix> cameras, ReplayFn replay,
                EndFrameCallback present, SubmitFn submit);

} // namespace aurora::gfx::interp
