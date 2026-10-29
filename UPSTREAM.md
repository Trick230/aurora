# Aurora fork: provenance

DualizeRecomp's fork of Aurora, on branch `dualize`. Changes are normal commits
here; upstream is merged by hand when it has something the fork wants.

- upstream: `https://github.com/encounter/aurora`
- fork point: `0549581` (Migrate remaining GX functions to FIFO, 2026-06-22)
- license: MIT (upstream `LICENSE` kept in this directory; keep the attribution)

## What the fork changes

- **Guest GX FIFO host** (`include/aurora/fifo.h`, `lib/gx/fifo.cpp`): a
  recompiled or emulated game's retail FIFO, rendered on a render thread of its
  own. The host queues its gather-pipe bursts, resolves guest addresses for
  display lists, vertex arrays, textures and TLUTs, gets the draw-done and
  token markers back, and saves and restores the decoder's state for
  savestates. `lib/gx/recomp.{cpp,hpp}` is the decoder's seam.
- **Frame interpolation** (`lib/gfx/frame_interp.cpp`): presents between the
  guest's frames, blending draws matched by object identity, camera and bones.
- **Image quality**: MSAA 4x with depth copies and alpha to coverage, render
  scale reduced to the window by an area filter, forced anisotropy.
- **Pipelines**: compiled on a worker pool, the cache precompiled at start,
  multisampled entries following the run's sample count.
- **Retail fixes**: XF channel control attenuation decode, TLUT loads that
  rewrite the same register, cull-all draws dropped, live EFB copies kept
  across a resize.
- **Stripped**: what DualizeRecomp never runs (the card, DVD and other
  library shims, mobile paths, the SDL_Renderer fallback, tests and examples)
  and the paths the GX stream never reaches; D3D12, Vulkan and Metal are the
  only backends, so pipelines always build on workers. ImGui idles until a
  menu is opened. Per-draw work is trimmed on the merge path.

## Upstream ported since the fork point

Checked against upstream `505b5eb` (2026-10-29):

- min filter decode (`26371ce`), packed TEV alpha comparisons (`3094e04`)
- range fog with its factors precomputed (`a59d20b` and earlier), plus the
  orthographic fog type
- EFB copies: alpha 1 from an EFB without alpha (`3840bf9`), clears of the
  copy rect only and `GXSetDstAlpha` through the shader (`9c0bf66`)
- no `tev_overflow` on registers a clamped stage wrote (`4998dcc`)
- texture reuse by contents (`7af8def`), on the FIFO's retail texture cache

Not taken: the RmlUi, input, WPAD, Android and THP work, the public gfx APIs,
and the regs/recording refactors the FIFO host replaces.
