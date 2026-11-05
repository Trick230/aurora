// Frame interpolation is not ported to the RHI yet: frames go straight to the screen.
#include "frame_interp.hpp"

namespace aurora::gfx::interp {
rhi::BindGroup* g_vertexUniformBindGroup = nullptr;
void initialize(bool, uint32_t) {}
void shutdown() {}
bool enabled() noexcept { return false; }
bool replaying() noexcept { return false; }
void discard() {}
void push_frame(ByteBuffer, std::vector<DrawRecord>, std::vector<ViewMatrix>, ReplayFn, EndFrameCallback, SubmitFn) {}
} // namespace aurora::gfx::interp
