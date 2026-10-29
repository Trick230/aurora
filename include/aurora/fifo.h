#pragma once

// A recompiled or emulated game's retail GX FIFO, rendered with Aurora on a
// render thread and window of its own. The game's CPU thread queues the bytes
// its gather pipe writes; the render thread decodes them (gx/recomp.hpp's
// process_stream), reports the draw-done and token markers back, and presents
// at each display copy.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Called on the render thread.
typedef struct AuroraFifoHost {
  void* user;
  // Guest address (virtual or physical) to host memory covering at least size
  // bytes; available receives the bytes left in that memory region.
  bool (*resolve)(void* user, uint32_t address, uint32_t size, const void** data,
                  uint32_t* available);
  // The GPU reached a GXSetDrawDone / GXSetDrawSync marker.
  void (*draw_done)(void* user);
  void (*token)(void* user, uint16_t token, bool interrupt);
  // The user closed the window.
  void (*quit)(void* user);
  // Optional: the render thread started (to name it for a profiler, say).
  void (*thread_started)(void* user);
} AuroraFifoHost;

typedef struct AuroraFifoConfig {
  const char* title;
  const char* cache_path;  // existing directory for the pipeline cache and settings
  // HD texture packs: DDS/PNG files named like Dolphin's dumps (tex1_...).
  const char* texture_path;
  uint32_t width;
  uint32_t height;
  bool vsync;
  // Present replays between the guest's frames, blended between the last two
  // (see frame_interp), at the refresh rate or present_rate (no vsync; 0 = the
  // refresh rate).
  bool interpolate;
  uint32_t present_rate;
  // The aspect the guest's frames are shown at; the window keeps it.
  float aspect;
  // Multisampling (1 or 4 samples) and the frame buffer's size over the
  // window's; above 1 the frame is reduced to the window by an area filter.
  uint32_t msaa;
  float render_scale;
  // Writes each texture no pack replaces to <cache>/texture_dumps.
  bool dump_textures;
  // One line per second of render-thread counters on stderr.
  bool stats;
} AuroraFifoConfig;

// Starts the render thread and waits until Aurora is up and every pipeline
// the cache knows is compiled. False on failure.
bool aurora_fifo_start(const AuroraFifoConfig* config, const AuroraFifoHost* host);
// Queues big-endian FIFO bytes (the guest's gather-pipe bursts). Producer
// thread only; blocks while the render thread is a full ring behind.
void aurora_fifo_submit(const uint8_t* data, size_t size);
// Producer thread only: returns once every queued byte has been processed and
// its draw-done and token markers reported, so emulator state saved afterwards
// holds them. Gives up after a few seconds if the render thread is stalled.
void aurora_fifo_drain(void);
// Producer thread only, after aurora_fifo_drain: the decoder state (GX
// registers and any incomplete packet) as bytes for aurora_fifo_load_state.
// The buffer stays valid until the next call.
const uint8_t* aurora_fifo_save_state(size_t* size);
// Producer thread only: replaces the decoder state with one saved by
// aurora_fifo_save_state; the stream submitted next continues from it.
void aurora_fifo_load_state(const uint8_t* data, size_t size);
void aurora_fifo_stop(void);

#ifdef __cplusplus
}
#endif
