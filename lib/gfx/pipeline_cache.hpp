#pragma once

#include "common.hpp"

#include <functional>

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

using NewPipelineCallback = std::function<std::shared_ptr<rhi::RenderPipeline>()>;

void initialize_pipeline_cache();
void shutdown_pipeline_cache();

template <typename Config>
PipelineRef find_pipeline(ShaderType type, const Config& config, NewPipelineCallback&& cb);
// Returns once the pipeline is ready, for a draw that has nothing else to draw with.
PipelineRef find_pipeline_blocking(ShaderType type, const gx::PipelineConfig& config, NewPipelineCallback&& cb);
PipelineRef find_pipeline_blocking(ShaderType type, const clear::PipelineConfig& config, NewPipelineCallback&& cb);

rhi::RenderPipeline* get_pipeline(PipelineRef ref);

} // namespace aurora::gfx
