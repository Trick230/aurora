#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stdint.h"
#endif

typedef struct {
  uint32_t queuedPipelines;
  uint32_t createdPipelines;
  uint32_t drawCallCount;
  uint32_t mergedDrawCallCount;
  uint32_t droppedDrawCallCount; // draws that did not fit the frame's staging buffers
  uint32_t lastVertSize;
  uint32_t lastUniformSize;
  uint32_t lastIndexSize;
  uint32_t lastStorageSize;
  uint32_t lastTextureUploadSize;
  // Frame interpolation, running totals: frames replayed, replays presented,
  // the frames' GX draws and those of them blended.
  uint32_t interpFrames;
  uint32_t interpPresents;
  uint32_t interpDraws;
  uint32_t interpBlendedDraws;
  uint32_t interpMicros; // render worker time spent blending and encoding replays
  // Hitches: replays that held a frame past its end because the next one was
  // late, and presents more than half an interval after the one before.
  uint32_t interpStarved;
  uint32_t interpLatePresents;
  // Running total of draws encoded while their pipeline was still compiling
  // (skipped: the pop-in).
  uint32_t skippedDraws;
  // Running total of draws recorded with the ubershader, their own pipeline
  // still compiling.
  uint32_t uberDraws;
} AuroraStats;

const AuroraStats* aurora_get_stats();
// GPU time of each command list, for stats: off by default, as it records timestamps.
void aurora_set_gpu_timing(bool on);
// GPU nanoseconds of the command lists finished since the last call; 0 while off.
uint64_t aurora_take_gpu_ns();

#ifdef __cplusplus
}
#endif

#endif
