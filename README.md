# Aurora (DualizeRecomp fork)

A renderer for the GX command stream of GameCube and Wii games. A recompiled
or emulated game writes its retail GX FIFO as it would on the console; Aurora
decodes it on a render thread of its own and draws it with D3D12, Vulkan or
Metal, in a window it owns, with frame interpolation, multisampling and
render scaling on top.

This is the fork used by [DualizeRecomp](https://github.com/Trick230/DualizeRecomp).
Upstream is [encounter/aurora](https://github.com/encounter/aurora), a
source-level compatibility layer for decompilation projects; this fork keeps
its GX renderer and drops the rest. `UPSTREAM.md` lists what the fork changes
and what it has taken from upstream since.

Which Aurora to use:

- **A static recompilation or an emulator** (the game's own code builds GX
  commands and writes them to the gather pipe): this fork. It needs nothing
  from the game but the bytes it writes and its memory.
- **A decompilation or source port** (the game's source calls `GXBegin`,
  `GXSetTevOp` and the rest of the SDK in C): upstream. The fork has no C GX
  API, PAD, DVD or CARD layers.

- [Features](#features)
- [Building](#building)
- [Quick start](#quick-start)
- [The FIFO host](#the-fifo-host)
- [Configuration](#configuration)
- [Frame interpolation](#frame-interpolation)
- [Pipeline cache](#pipeline-cache)
- [Textures](#textures)
- [Savestates](#savestates)
- [Statistics](#statistics)
- [Limitations](#limitations)
- [License](#license)

## Features

**The stream**

- Decodes the retail FIFO as the hardware command processor does: CP, XF and
  BP register writes, draws, display lists, indexed and inline matrix loads.
  Display lists, vertex arrays, textures and TLUTs are read from guest memory
  through a resolver the host provides.
- Reports what the guest waits for: `GXSetDrawDone`, draw syncs and tokens
  (with or without their interrupt) and each display copy, which ends a frame.
- A lock-free 16 MB ring between the game's CPU thread and the render thread;
  the CPU thread waits only when the renderer is a full ring behind.

**Rendering**

- D3D12 and Vulkan on Windows, Vulkan on Linux, Metal on macOS, through
  [Dawn](https://dawn.googlesource.com/dawn/) (WebGPU).
- Every TEV, texgen, lighting, fog and blend setup compiled to its own
  pipeline, on a worker pool (the cores less two).
- EFB copies (shadows, bloom, reflections, render-to-texture) as GPU
  textures, scaled with the render resolution, kept per destination address
  in an LRU of 64 MB.
- Range and orthographic fog, destination alpha, partial clears, clamped TEV
  stages and the other retail details listed in `UPSTREAM.md`.

**Image quality**

- MSAA 4x, with depth copies resolved from sample 0.
- Alpha to coverage under MSAA for unblended alpha-test cutouts, where the
  EFB keeps no alpha.
- A render scale above the window's size, reduced with an area (box) filter
  so every rendered pixel counts.
- 16x anisotropic filtering on every mipmapped texture.
- 8 bits a channel with no dithering; mip selection follows the render
  resolution.
- A display aspect (4:3, 16:9) kept as the window is resized, letterboxed.

**Frame interpolation**

- Presents between the game's frames at the display's refresh rate (or a set
  rate without vsync), each draw's matrices blended toward the same draw one
  frame earlier. Draws are matched by their contents and matrices by where the
  game loaded them from, so skinned models blend bone by bone.
- A game that names its camera (one host packet in the stream) gets its
  objects blended in the world and the camera moved along its own path.

**Startup and assets**

- A pipeline cache saved between runs, a seed cache shipped with a release
  and merged in at each start, and everything the cache knows compiled before
  the game starts, so none of it pops in.
- HD texture packs in Dolphin's naming (`tex1_...`, DDS or PNG), and texture
  dumps to make them.

## Building

Requirements: CMake 3.25+, a C++20 compiler (Clang, GCC or MSVC) and Git.
Dawn and SDL3 are downloaded as prebuilt packages by default.

Add the fork as a subdirectory and link its two libraries:

```cmake
add_subdirectory(thirdparty/aurora EXCLUDE_FROM_ALL)
target_link_libraries(your_runtime PRIVATE aurora::core aurora::gx)
```

| Option | Values | Default |
|---|---|---|
| `AURORA_DAWN_PROVIDER` | `auto`, `package` (prebuilt), `vendor` (from source), `system` | `auto` |
| `AURORA_DAWN_LINKAGE` | `shared`, `static` | `shared` on Windows, else `static` |
| `AURORA_SDL3_PROVIDER` | as for Dawn | `auto` |
| `AURORA_SDL3_LINKAGE` | `shared`, `static` | as for Dawn |
| `AURORA_DAWN_VERSION`, `AURORA_SDL3_VERSION` | the prebuilt release to download | pinned in `CMakeLists.txt` |

With a shared Dawn and SDL3, ship `webgpu_dawn` and `SDL3` (`.dll`, `.so` or
`.dylib`) next to the executable; on Windows also `dxcompiler.dll` and
`dxil.dll`, which compile the D3D12 shaders.

## Quick start

The whole interface is `include/aurora/fifo.h`. On the game's CPU thread:

```c
#include <aurora/fifo.h>

static bool resolve(void* user, uint32_t address, uint32_t size,
                    const void** data, uint32_t* available) {
  /* Map a guest address (virtual or physical, MEM1 or MEM2) to host memory. */
  uint32_t phys = address & 0x1FFFFFFF;
  if (phys + size > MEM1_SIZE)
    return false;
  *data = mem1 + phys;
  *available = MEM1_SIZE - phys;
  return true;
}

static void draw_done(void* user) { /* raise PE finish for the guest */ }
static void token(void* user, uint16_t value, bool interrupt) { /* PE token */ }
static void quit(void* user) { /* the window was closed */ }

const AuroraFifoConfig config = {
  .title = "My Game",
  .cache_path = "cache",          /* must exist */
  .texture_path = "textures",     /* may be empty or missing */
  .width = 1280, .height = 720,
  .vsync = true,
  .interpolate = true,
  .aspect = 16.0f / 9.0f,
  .msaa = 4,
  .render_scale = 1.0f,
};
const AuroraFifoHost host = {
  .resolve = resolve, .draw_done = draw_done, .token = token, .quit = quit,
};
if (!aurora_fifo_start(&config, &host))
  return error;

/* Each time the game's write-gather pipe fills (32-byte bursts): */
aurora_fifo_submit(burst, 32);

/* At shutdown: */
aurora_fifo_stop();
```

`aurora_fifo_start` opens the window, compiles the cached pipelines and
returns once the render thread is ready. From then on the renderer runs on
its own: it presents at each display copy and polls the window's events.

## The FIFO host

`AuroraFifoHost` is how Aurora reaches back into the game. Every callback
runs on the render thread.

| Callback | When | What the host does |
|---|---|---|
| `resolve` | Aurora reads guest memory: a display list, vertex array, texture or TLUT | Returns a pointer to host memory covering at least `size` bytes at `address` (virtual or physical), and in `available` how many bytes that region holds from there. False for an address outside RAM |
| `draw_done` | The stream reached a `GXSetDrawDone` marker | Raises the PE finish interrupt |
| `token` | The stream reached a draw sync token | Sets the PE token register; raises the token interrupt if `interrupt` |
| `quit` | The user closed the window | Stops the game |
| `thread_started` | Optional: the render thread started | Names the thread for a profiler, say |

**Determinism.** Those callbacks arrive whenever the render thread gets to
them, which depends on how far behind the CPU thread it is. A host that wants
the game to run the same way every time records them and delivers them to the
guest only at points of the guest's own clock: call `aurora_fifo_drain` there
(when the CPU goes idle, at the end of each video field, when the guest reads
the CP or PE registers), and once it returns every marker submitted before is
reported.

| Function | Thread | Does |
|---|---|---|
| `aurora_fifo_start` | Any, once | Starts the render thread; false on failure |
| `aurora_fifo_submit` | Producer | Queues big-endian FIFO bytes; blocks only while the ring is full |
| `aurora_fifo_drain` | Producer | Returns once everything queued is processed and its markers reported; gives up after a few seconds if the renderer is stalled |
| `aurora_fifo_save_state` | Producer, after a drain | The decoder's state as bytes (see [Savestates](#savestates)) |
| `aurora_fifo_load_state` | Producer | Restores a saved state; the next bytes continue from it |
| `aurora_fifo_stop` | Producer | Stops the render thread and closes the window |

The producer is the one thread that submits: the game's CPU thread.

## Configuration

`AuroraFifoConfig`; the strings are copied, so they need not outlive the call.

| Field | Meaning |
|---|---|
| `title` | Window title |
| `cache_path` | An existing directory for the pipeline cache, Dawn's shader cache and settings |
| `texture_path` | A directory of HD texture packs (see [Textures](#textures)) |
| `width`, `height` | The window's starting size |
| `vsync` | Present on the display's refresh |
| `interpolate` | Frame interpolation (see below) |
| `present_rate` | Presents per second without vsync; 0 means the display's refresh rate |
| `aspect` | The aspect the game's frames are shown at (4:3, 16:9); 0 stretches them to the window |
| `msaa` | 1 or 4 samples (WebGPU allows no other count) |
| `render_scale` | The frame buffer's size over the window's: 1 renders at the window's size, above it renders larger and reduces with an area filter |
| `dump_textures` | Writes each texture no pack replaces to `<cache_path>/texture_dumps` |
| `stats` | One line of counters per second on stderr (see [Statistics](#statistics)) |

## Frame interpolation

A GX game renders at its own rate, most often 30 FPS. With `interpolate` on,
Aurora shows each game frame one frame late and, until the next one arrives,
presents replays of it with every draw's matrices blended toward the matching
draw of the frame before. Each replay sits between two frames the game really
produced, so nothing is predicted.

- Draws are matched across frames by their pipeline, textures and sizes, and
  their order among draws alike. Within a draw, matrices are paired by where
  the game loaded them from, since a skinned model's bones can land in
  different matrix slots each frame.
- A draw with no match, or one whose matrices moved too far to be the same
  object (a camera cut), is drawn as the newer frame has it.
- Texture matrices blend only while they move a little: a scrolling
  texture's offset wraps.
- Lights and post-transform texture matrices (projected shadows) move with
  the camera.

**Naming the camera.** On its own Aurora estimates the camera's move from
the draws and guesses which objects stand in the world and which follow the
camera. A game can say instead: a hook where the game computes its view
writes one host packet into the gather pipe, ahead of the draws built with
that camera. The draws after it, up to the next packet or the frame's end,
are then blended in the world and seen through the camera's own move.

```
0x70                         opcode (no retail packet has these bits)
12 x big-endian float        the view matrix, row-major 3x4, world to eye space
```

Many Wii games use NintendoWare's g3d, whose `ScnRoot::CalcView` computes the
view of every scene; DualizeRecomp's `sdk/nw4r_g3d.c` hooks it, and the same
hook serves any g3d game. A game with its own engine needs a hook of its own.

Interpolation needs the game's frames to arrive regularly and the throttle
on: run uncapped, there is nothing to present between.

## Pipeline cache

Each GX setup becomes a GPU pipeline, compiled the first time a draw needs
it. Aurora saves what it compiled to `<cache_path>/pipeline_cache.db`, and at
the next start compiles all of it on the worker pool before the game starts.

A release can ship a seed: `initial_pipeline_cache.db` next to the
executable, merged into the user's cache at each start. Build it by playing
the game through and taking the resulting `pipeline_cache.db`
(DualizeRecomp's `scripts/pipeline_seed.py` merges several). A seed made with
MSAA on serves MSAA off too.

A pipeline the cache has never seen still compiles mid-play, and its draws
are skipped until it is ready (the stats line counts them). Changes to the
pipeline key (`GXPipelineConfigVersion` in `lib/gx/pipeline.hpp`) invalidate
every cache and seed, so bump it only when the layout really changes.

## Textures

Textures are decoded from guest memory and cached by their contents, so a
texture the game rewrites at the same address is noticed.

`texture_path` holds replacement packs in Dolphin's naming:
`tex1_<width>x<height>_<hash>_<format>.dds` (or `.png`), with the TLUT's hash
for paletted textures. Packs made for Dolphin work unchanged. With
`dump_textures` on, every texture no pack replaces is written to
`<cache_path>/texture_dumps` under the name a replacement would take.

## Savestates

The renderer's part of a savestate is the decoder's state: the last value
written to every GX register, and any packet the game had only half written.

```c
aurora_fifo_drain();
size_t size;
const uint8_t* gx = aurora_fifo_save_state(&size);   /* store size bytes */
...
aurora_fifo_drain();
aurora_fifo_load_state(gx, size);                    /* then resume submitting */
```

Restoring replays the registers as FIFO packets, so the next draw is set up as
it was. Textures, EFB copies and pipelines are rebuilt from guest memory and
the cache as the game draws.

## Statistics

With `stats` on, a line a second on stderr:

| Counter | Meaning |
|---|---|
| `fps`, `draws (dropped)` | Frames, draws, draws that did not fit the frame's buffers |
| `MB fifo`, `draw-done`, `tokens` | Stream bytes and markers |
| `stalled frame starts` | Frames that waited for a free frame slot |
| `peak MB uniforms / vertices / arrays` | The largest frame's uploads |
| `ms decode / begin / end / idle` | Render thread time: decoding, waiting for a frame slot, handing the frame over, waiting for the game |
| `cpu waits` | Time the game's thread waited on a full ring or a drain |
| `frame ms`, `sd`, `max` | Intervals between the game's frames |
| `interpolated presents` | Replays shown, the share of draws blended, the time each took, and the hitches (starved, late) |
| `pipelines compiled`, `draws skipped compiling` | Pipelines built this second, and draws skipped for one still compiling |

`aurora_get_stats()` (`include/aurora/gfx.h`) returns the running totals.

## Limitations

- **One renderer per process.** The FIFO host is a single instance.
- **New pipelines pop in.** A draw whose pipeline is still compiling is
  skipped for those frames. The cache and seed cover what has been played; a
  pipeline nothing has seen still compiles mid-play.
- **MSAA is 1x or 4x.** WebGPU allows no other sample count.
- **EFB copies stay on the GPU.** Textures the game samples from a copy's
  address use the copy, but its pixels never reach guest memory, so a game
  that reads them on the CPU sees what was there before.
- **No C GX API.** The fork renders the FIFO only; decompilation projects
  should use upstream.
- **Tested on Windows** (D3D12 and Vulkan). Linux and macOS have the code
  paths (Vulkan, Metal, a portable frame pacer) but no testing yet.

## License

MIT, as upstream (`LICENSE`).
