#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
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
} AuroraStats;

const AuroraStats* aurora_get_stats();

#ifdef __cplusplus
}
#endif

#endif
