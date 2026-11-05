#pragma once

#include "../gfx/refs.hpp"

#include <dolphin/gx/GXEnum.h>

#include <cstdint>

// What a draw is, on either side of the renderer and with no graphics API in it.
namespace aurora::gx {
// A draw's vertices in the game's vertex format, wherever they came from. Everything
// else it draws with is in g_gxState, whoever set it: the FIFO's register writes or
// a call to GX.
struct VertexPacket {
  GXPrimitive prim;
  GXVtxFmt fmt;
  uint16_t vtxCount;
  const uint8_t* vertices;
  bool bigEndian;
};
void submit_draw(const VertexPacket& packet);

struct GXBindGroups {
  gfx::BindGroupRef textureBindGroup;
};

// What the renderer gets for it: the ranges are in the frame's buffers, the references
// in the renderer's caches.
struct DrawData {
  gfx::PipelineRef pipeline;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range uniformRange;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t firstInstance; // vertex byte offset for single-instance draws (vtx_start)
  GXBindGroups bindGroups;
  uint32_t dstAlpha;
  GXCullMode cullMode;
};
} // namespace aurora::gx
