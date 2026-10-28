#ifndef AURORA_EVENT_H
#define AURORA_EVENT_H

#include "aurora.h"

#include <SDL3/SDL_events.h>

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
#include "stdint.h"
#endif

typedef enum {
  AURORA_NONE,
  AURORA_EXIT,
  AURORA_SDL_EVENT,
} AuroraEventType;

struct AuroraEvent {
  AuroraEventType type;
  SDL_Event sdl;
};

#ifdef __cplusplus
}
#endif

#endif
