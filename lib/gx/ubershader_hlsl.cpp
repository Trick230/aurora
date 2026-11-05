// The ubershader in HLSL, function by function as UberSource in ubershader.cpp: it
// must draw the very pixels the generated shader does, and a change to one goes in
// the others.
#include "shader_hlsl.hpp"

namespace aurora::gx::hlsl {
std::string_view uber_body() noexcept {
  return R"""(
struct Light {
    float3 pos;
    float3 dir;
    float4 color;
    float3 cos_att;
    float3 dist_att;
};
struct Fog {
    float4 color;
    float a;
    float b;
    float c;
    float pad;
};
struct Uniform {
    uint vtx_start;
    uint current_pnmtx;
    float2 render_viewport_size;
    float2 logical_viewport_size;
    uint fog_range_base;
    uint config_base;
    uint4 array_start[3];
    float4x4 proj;
    float4x3 postex_mtx[20];
    float4x3 nrm_mtx[10];
    float4 tev_regs[4];
    Light lights[8];
    uint4 light_state;
    float4 cc_amb[4];
    float4 cc_mat[4];
    float4 kcolors[4];
    float4x3 postmtx[20];
    Fog fog;
    float4 texcoord_scale[8];
    float4 ind_mtx[6];
    float4 tex_size_bias[8];
};
ByteAddressBuffer vbuf : register(t0, space0);
ByteAddressBuffer abuf : register(t1, space0);
ConstantBuffer<Uniform> ubuf : register(b0, space1);
Texture2D tex0 : register(t0, space2);
SamplerState tex0_samp : register(s1, space2);
Texture2D tex1 : register(t2, space2);
SamplerState tex1_samp : register(s3, space2);
Texture2D tex2 : register(t4, space2);
SamplerState tex2_samp : register(s5, space2);
Texture2D tex3 : register(t6, space2);
SamplerState tex3_samp : register(s7, space2);
Texture2D tex4 : register(t8, space2);
SamplerState tex4_samp : register(s9, space2);
Texture2D tex5 : register(t10, space2);
SamplerState tex5_samp : register(s11, space2);
Texture2D tex6 : register(t12, space2);
SamplerState tex6_samp : register(s13, space2);
Texture2D tex7 : register(t14, space2);
SamplerState tex7_samp : register(s15, space2);

uint cfg(uint word) {
  return abuf.Load((ubuf.config_base + word) * 4u);
}
float cfg_f32(uint word) {
  return asfloat(cfg(word));
}
uint byte_of(uint value, uint index) {
  return (value >> (index * 8u)) & 0xFFu;
}
uint nibble_of(uint value, uint index) {
  return (value >> (index * 4u)) & 0xFu;
}

// ---------------------------------------------------------------- vertices

static const uint VA_PNMTXIDX = 0u;
static const uint VA_TEX0MTXIDX = 1u;
static const uint VA_POS = 9u;
static const uint VA_NRM = 10u;
static const uint VA_CLR0 = 11u;
static const uint VA_TEX0 = 13u;
static const uint AT_NONE = 0u;
static const uint AT_INDEX8 = 2u;
static const uint AT_INDEX16 = 3u;

static uint vtx_start_of_draw;

struct AttrAddress {
  uint offs;
  bool in_arrays;
  bool le;
};

AttrAddress make_address(uint offs, bool in_arrays, bool le) {
  AttrAddress result;
  result.offs = offs;
  result.in_arrays = in_arrays;
  result.le = le;
  return result;
}

uint attr_type(uint attr) {
  return cfg(CFG_ATTRS + attr * 2u) & 0xFFu;
}

AttrAddress attr_address(uint attr, uint vidx, uint dl_extra, uint within) {
  uint w0 = cfg(CFG_ATTRS + attr * 2u);
  uint w1 = cfg(CFG_ATTRS + attr * 2u + 1u);
  uint type_ = byte_of(w0, 0u);
  uint entry = vtx_start_of_draw + vidx * cfg(CFG_VTX_STRIDE) + byte_of(w0, 3u) + dl_extra;
  uint stride = byte_of(w1, 0u);
  bool le = byte_of(w1, 2u) != 0u;
  if (type_ == AT_INDEX8) {
    return make_address(ARRAY_START(attr - VA_POS) + raw_fetch_u8_1(vbuf, entry) * stride + within, true, le);
  }
  if (type_ == AT_INDEX16) {
    return make_address(ARRAY_START(attr - VA_POS) + raw_fetch_u16_1(vbuf, entry, false) * stride + within, true, le);
  }
  return make_address(entry + within, false, false);
}

float fetch_component(ByteAddressBuffer p, uint comp, uint offs, uint fraction, bool le) {
  switch (comp) {
    case 0u: { return fetch_u8_1(p, offs, fraction, le); }
    case 1u: { return fetch_s8_1(p, offs, fraction, le); }
    case 2u: { return fetch_u16_1(p, offs, fraction, le); }
    case 3u: { return fetch_s16_1(p, offs, fraction, le); }
    default: { return fetch_f32_1(p, offs, le); }
  }
}

uint component_size(uint comp) {
  if (comp <= 1u) { return 1u; }
  if (comp <= 3u) { return 2u; }
  return 4u;
}

float3 fetch_vector(ByteAddressBuffer p, uint comp, uint count, uint offs, uint fraction, bool le) {
  float3 v = float3(0.0, 0.0, 0.0);
  uint size = component_size(comp);
  for (uint i = 0u; i < min(count, 3u); i++) {
    v[i] = fetch_component(p, comp, offs + i * size, fraction, le);
  }
  return v;
}

float3 load_vector(uint attr, uint count, uint vidx, uint dl_extra, uint within) {
  uint w0 = cfg(CFG_ATTRS + attr * 2u);
  uint fraction = byte_of(cfg(CFG_ATTRS + attr * 2u + 1u), 1u);
  uint comp = byte_of(w0, 2u);
  AttrAddress a = attr_address(attr, vidx, dl_extra, within);
  if (a.in_arrays) {
    return fetch_vector(abuf, comp, count, a.offs, fraction, a.le);
  }
  return fetch_vector(vbuf, comp, count, a.offs, fraction, a.le);
}

float3 load_attr(uint attr, uint vidx) {
  return load_vector(attr, byte_of(cfg(CFG_ATTRS + attr * 2u), 1u), vidx, 0u, 0u);
}

float3 load_pos(uint vidx) {
  return load_attr(VA_POS, vidx);
}

float3 load_nrm(uint vidx) {
  if (attr_type(VA_NRM) == AT_NONE) {
    return float3(1.0, 0.0, 0.0);
  }
  return load_attr(VA_NRM, vidx);
}

// The binormal (1) or tangent (2) of a normal with all three.
float3 load_nbt_slice(uint slice, uint vidx) {
  uint w0 = cfg(CFG_ATTRS + VA_NRM * 2u);
  uint w1 = cfg(CFG_ATTRS + VA_NRM * 2u + 1u);
  uint type_ = byte_of(w0, 0u);
  uint dl_extra = 0u;
  if (byte_of(w1, 3u) != 0u) {
    if (type_ == AT_INDEX8) {
      dl_extra = slice;
    } else if (type_ == AT_INDEX16) {
      dl_extra = slice * 2u;
    }
  }
  return load_vector(VA_NRM, 3u, vidx, dl_extra, slice * 3u * component_size(byte_of(w0, 2u)));
}

uint load_index_byte(uint attr, uint vidx) {
  AttrAddress a = attr_address(attr, vidx, 0u, 0u);
  if (a.in_arrays) {
    return raw_fetch_u8_1(abuf, a.offs);
  }
  return raw_fetch_u8_1(vbuf, a.offs);
}

uint load_pnmtx(uint vidx) {
  if (attr_type(VA_PNMTXIDX) == AT_NONE) {
    return ubuf.current_pnmtx;
  }
  return load_index_byte(VA_PNMTXIDX, vidx) / 3u;
}

float4 fetch_color(ByteAddressBuffer p, uint comp, uint offs, bool le) {
  switch (comp) {
    case 0u: { return fetch_rgb565(p, offs, le); }
    case 1u: { return fetch_rgb8(p, offs, le); }
    case 2u: { return fetch_rgbx8(p, offs, le); }
    case 3u: { return fetch_rgba4(p, offs, le); }
    case 4u: { return fetch_rgba6(p, offs, le); }
    default: { return fetch_rgba8(p, offs, le); }
  }
}

float4 load_color(uint channel, uint vidx) {
  uint attr = VA_CLR0 + channel;
  if (attr_type(attr) == AT_NONE) {
    return vec4f(0.0);
  }
  uint comp = byte_of(cfg(CFG_ATTRS + attr * 2u), 2u);
  AttrAddress a = attr_address(attr, vidx, 0u, 0u);
  if (a.in_arrays) {
    return fetch_color(abuf, comp, a.offs, a.le);
  }
  return fetch_color(vbuf, comp, a.offs, a.le);
}

// ---------------------------------------------------------------- lighting

float4 light_channel(uint index, float4 vertex_color, float3 mv_pos, float3 mv_nrm) {
  uint config = cfg(CFG_CHANNELS + index);
  float4 material = select(ubuf.cc_mat[index], vertex_color, nibble_of(config, 0u) == 1u);
  if (nibble_of(config, 4u) == 0u) {
    return material;
  }
  uint diff_fn = nibble_of(config, 2u);
  uint attn_fn = nibble_of(config, 3u);
  float4 lighting = select(ubuf.cc_amb[index], vertex_color, nibble_of(config, 1u) == 1u);
  for (uint i = 0u; i < 8u; i++) {
    if ((ubuf.light_state[index] & (1u << i)) == 0u) { continue; }
    Light light = ubuf.lights[i];
    float3 ldir = light.pos - mv_pos;
    float dist2 = dot(ldir, ldir);
    float dist = sqrt(dist2);
    ldir = ldir / dist;
    float attn = 1.0;
    if (attn_fn == 1u) { // spot
      float cosine = max(0.0, dot(ldir, light.dir));
      float cos_attn = dot(light.cos_att, vec3f(1.0, cosine, cosine * cosine));
      float dist_attn = dot(light.dist_att, vec3f(1.0, dist, dist2));
      attn = max(0.0, cos_attn / dist_attn);
    } else if (attn_fn == 0u) { // specular
      attn = select(0.0, max(0.0, dot(mv_nrm, light.dir)), dot(mv_nrm, ldir) >= 0.0);
      float cos_attn = dot(light.cos_att, vec3f(1.0, attn, attn * attn));
      float dist_attn;
      if (diff_fn != 0u) {
        dist_attn = max(0.0, dot(normalize(light.dist_att), vec3f(1.0, attn, attn * attn)));
      } else {
        dist_attn = max(0.0, dot(light.dist_att, vec3f(1.0, attn, attn * attn)));
      }
      attn = max(0.0, cos_attn / dist_attn);
    }
    float diff = 1.0;
    if (diff_fn == 1u) {
      diff = dot(ldir, mv_nrm);
    } else if (diff_fn == 2u) {
      diff = max(0.0, dot(ldir, mv_nrm));
    }
    lighting = lighting + (attn * diff * light.color);
  }
  return material * clamp(lighting, vec4f(0.0), vec4f(1.0));
}

// ---------------------------------------------------------------- vertex stage

static const uint TG_MTX3X4 = 0u;
static const uint TG_MTX2X4 = 1u;
static const uint TG_BUMP0 = 2u;
static const uint TG_BUMP7 = 9u;
static const uint TG_SRTG = 10u;
static const uint TG_POS = 0u;
static const uint TG_NRM = 1u;
static const uint TG_BINRM = 2u;
static const uint TG_TANGENT = 3u;
static const uint TG_TEX0 = 4u;
static const uint TG_TEX7 = 11u;
static const uint TG_COLOR0 = 19u;

struct VertexOutput {
    float4 pos : SV_Position;
    float4 cc0 : TEXCOORD0;
    float4 cc1 : TEXCOORD1;
    float3 tex0 : TEXCOORD2;
    float3 tex1 : TEXCOORD3;
    float3 tex2 : TEXCOORD4;
    float3 tex3 : TEXCOORD5;
    float3 tex4 : TEXCOORD6;
    float3 tex5 : TEXCOORD7;
    float3 tex6 : TEXCOORD8;
    float3 tex7 : TEXCOORD9;
};

VertexOutput vs_main(uint vidx : SV_VertexID, uint iidx : SV_InstanceID) {
    VertexOutput output = (VertexOutput)0;
    uint line_mode = cfg(CFG_LINE_MODE);
    // Single-instance draws carry their vertex byte offset as a draw constant.
    vtx_start_of_draw = select(draw.vtx_start, ubuf.vtx_start, line_mode != 0u);
    uint in_vidx = vidx;
    float3 in_pos;
    uint in_pnmtxidx;
    float3 mv_pos;
    if (line_mode == 0u) {
        in_pos = load_pos(vidx);
        in_pnmtxidx = load_pnmtx(vidx);
        mv_pos = mul(vec4f(in_pos, 1.0), ubuf.postex_mtx[in_pnmtxidx]);
        output.pos = mul(vec4f(mv_pos, 1.0), ubuf.proj);
    } else if (line_mode == 3u) {
        // GX_POINTS: expand a vertex to an axis-aligned screen-space square.
        in_vidx = iidx;
        in_pos = load_pos(in_vidx);
        in_pnmtxidx = load_pnmtx(in_vidx);
        mv_pos = mul(vec4f(in_pos, 1.0), ubuf.postex_mtx[in_pnmtxidx]);
        float4 clip_pos = mul(vec4f(mv_pos, 1.0), ubuf.proj);
        float2 viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));
        float point_size = cfg_f32(CFG_LINE_WIDTH) * min(viewport_scale.x, viewport_scale.y);
        float x_sign = select(-1.0, 1.0, (vidx & 1u) != 0u);
        float y_sign = select(-1.0, 1.0, vidx >= 2u);
        float2 offset_px = vec2f(x_sign, y_sign) * (point_size / 2.0);
        float2 offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;
        output.pos = vec4f(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);
    } else {
        // GX_LINES, GX_LINESTRIP: expand a segment perpendicular to its direction.
        bool use_b = vidx >= 2u;
        uint vidx_a = iidx * select(1u, 2u, line_mode == 1u);
        uint vidx_b = vidx_a + 1u;
        in_vidx = select(vidx_a, vidx_b, use_b);
        float3 pos_a = load_pos(vidx_a);
        float3 pos_b = load_pos(vidx_b);
        in_pos = select(pos_a, pos_b, use_b);
        uint pnmtxidx_a = load_pnmtx(vidx_a);
        uint pnmtxidx_b = load_pnmtx(vidx_b);
        in_pnmtxidx = select(pnmtxidx_a, pnmtxidx_b, use_b);
        float3 mv_pos_a = mul(vec4f(pos_a, 1.0), ubuf.postex_mtx[pnmtxidx_a]);
        float3 mv_pos_b = mul(vec4f(pos_b, 1.0), ubuf.postex_mtx[pnmtxidx_b]);
        mv_pos = select(mv_pos_a, mv_pos_b, use_b);
        float4 clip_a = mul(vec4f(mv_pos_a, 1.0), ubuf.proj);
        float4 clip_b = mul(vec4f(mv_pos_b, 1.0), ubuf.proj);
        float2 ndc_a = clip_a.xy / clip_a.w;
        float2 ndc_b = clip_b.xy / clip_b.w;
        float2 viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));
        float2 delta_px = (ndc_b - ndc_a) / 2.0 * ubuf.render_viewport_size;
        float2 dir_px = select(vec2f(1.0, 0.0), normalize(delta_px), dot(delta_px, delta_px) > 1e-10);
        float2 perp_px = vec2f(-dir_px.y, dir_px.x);
        float line_width = cfg_f32(CFG_LINE_WIDTH) * min(viewport_scale.x, viewport_scale.y);
        float2 offset_px = perp_px * (line_width / 2.0) * select(-1.0, 1.0, (vidx & 1u) != 0u);
        float2 offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;
        float4 clip_base = select(clip_a, clip_b, use_b);
        output.pos = vec4f(clip_base.xy + offset_ndc * clip_base.w, clip_base.zw);
    }
    output.pos.z = -output.pos.z * (1.0 - 1e-7);

    float3 in_nrm = load_nrm(in_vidx);
    float3 nrm_tmp = mul(vec4f(in_nrm, 0.0), ubuf.nrm_mtx[in_pnmtxidx]);
    float3 mv_nrm = select(nrm_tmp, normalize(nrm_tmp), dot(nrm_tmp, nrm_tmp) > 1e-10);

    uint sampled_channels = cfg(CFG_SAMPLED_CHANNELS);
    if ((sampled_channels & 1u) != 0u) {
        float4 color = load_color(0u, in_vidx);
        output.cc0 = light_channel(0u, color, mv_pos, mv_nrm);
        output.cc0.a = light_channel(2u, color, mv_pos, mv_nrm).a;
    }
    if ((sampled_channels & 2u) != 0u) {
        float4 color = load_color(1u, in_vidx);
        output.cc1 = light_channel(1u, color, mv_pos, mv_nrm);
        output.cc1.a = light_channel(3u, color, mv_pos, mv_nrm).a;
    }

    float3 tc_out[8];
    for (uint i = 0u; i < 8u; i++) {
        tc_out[i] = float3(0.0, 0.0, 0.0);
    }
    uint sampled_coords = cfg(CFG_SAMPLED_COORDS);
    for (uint i = 0u; i < 8u; i++) {
        if ((sampled_coords & (1u << i)) == 0u) { continue; }
        uint w0 = cfg(CFG_TCGS + i * 2u);
        uint w1 = cfg(CFG_TCGS + i * 2u + 1u);
        uint type_ = byte_of(w0, 0u);
        uint src = byte_of(w0, 1u);
        if (type_ >= TG_BUMP0 && type_ <= TG_BUMP7) {
            // Emboss: the source texcoord, moved by the light across the surface.
            float3 ldir = normalize(ubuf.lights[type_ - TG_BUMP0].pos - mv_pos);
            float3 tangent = mul(vec4f(load_nbt_slice(2u, in_vidx), 0.0), ubuf.nrm_mtx[in_pnmtxidx]);
            float3 binormal = mul(vec4f(load_nbt_slice(1u, in_vidx), 0.0), ubuf.nrm_mtx[in_pnmtxidx]);
            float2 uv = tc_out[byte_of(w0, 2u)].xy + vec2f(dot(ldir, tangent), dot(ldir, binormal));
            tc_out[i] = vec3f(uv, 1.0);
            continue;
        }
        float4 tc;
        if (src >= TG_TEX0 && src <= TG_TEX7) {
            tc = vec4f(load_attr(VA_TEX0 + (src - TG_TEX0), in_vidx).xy, 1.0, 1.0);
        } else if (src == TG_POS) {
            tc = vec4f(in_pos, 1.0);
        } else if (src == TG_NRM) {
            tc = vec4f(in_nrm, 1.0);
        } else if (src >= TG_COLOR0) {
            // XF texgen sees the channel after lighting, not the vertex color.
            tc = select(output.cc0, output.cc1, src != TG_COLOR0);
        } else if (src == TG_BINRM) {
            tc = vec4f(load_nbt_slice(1u, in_vidx), 1.0);
        } else {
            tc = vec4f(load_nbt_slice(2u, in_vidx), 1.0);
        }
        float3 tmp;
        if (type_ == TG_SRTG) {
            tmp = vec3f(tc.xy, 1.0);
        } else {
            if (((w0 >> 25u) & 1u) != 0u) {
                tmp = mul(tc, ubuf.postex_mtx[load_index_byte(VA_TEX0MTXIDX + i, in_vidx) / 3u]);
            } else if (byte_of(w1, 0u) == 0xFFu) {
                tmp = tc.xyz;
            } else {
                tmp = mul(tc, ubuf.postex_mtx[byte_of(w1, 0u)]);
            }
            if (type_ == TG_MTX2X4) {
                tmp.z = 1.0;
            }
        }
        if (((w0 >> 24u) & 1u) != 0u) {
            tmp = normalize(tmp);
        }
        float3 proj = tmp;
        if (byte_of(w1, 1u) != 0xFFu) {
            proj = mul(vec4f(tmp, 1.0), ubuf.postmtx[byte_of(w1, 1u)]);
        }
        if (line_mode != 0u && (cfg(CFG_LINE_TEX_MASK) & (1u << i)) != 0u) {
            float tex_offset = cfg_f32(CFG_LINE_TEX_OFFSET);
            if (line_mode == 3u) {
                if ((vidx & 1u) != 0u) { proj.x += tex_offset; }
                if (vidx >= 2u) { proj.y += tex_offset; }
            } else if ((vidx & 1u) != 0u) {
                proj.y += tex_offset;
            }
        }
        tc_out[i] = proj;
    }
    output.tex0 = tc_out[0];
    output.tex1 = tc_out[1];
    output.tex2 = tc_out[2];
    output.tex3 = tc_out[3];
    output.tex4 = tc_out[4];
    output.tex5 = tc_out[5];
    output.tex6 = tc_out[6];
    output.tex7 = tc_out[7];
    return output;
}

// ---------------------------------------------------------------- fragment stage

static const uint FLAG_COLOR_CLAMP = 1u << 0u;
static const uint FLAG_ALPHA_CLAMP = 1u << 1u;
static const uint FLAG_COLOR_OVERFLOW_A = 1u << 2u;
static const uint FLAG_ALPHA_OVERFLOW_A = 1u << 5u;
static const uint FLAG_NEEDS_TEV_COORD = 1u << 8u;
static const uint FLAG_NEEDS_SAMPLE = 1u << 9u;
static const uint FLAG_HAS_BASE_TEXTURE = 1u << 10u;
static const uint FLAG_HAS_BASE_COORD = 1u << 11u;
static const uint FLAG_SIMPLE_COORDS = 1u << 12u;
static const uint FLAG_ADD_PREVIOUS = 1u << 13u;
static const uint FLAG_WRITES_COORD = 1u << 14u;
static const uint FLAG_ALPHA_COMPARES_COLOR = 1u << 15u;

float4 sample_map(uint map, float2 uv, float bias) {
  switch (map) {
    case 0u: { return textureSampleBias(tex0, tex0_samp, uv, bias); }
    case 1u: { return textureSampleBias(tex1, tex1_samp, uv, bias); }
    case 2u: { return textureSampleBias(tex2, tex2_samp, uv, bias); }
    case 3u: { return textureSampleBias(tex3, tex3_samp, uv, bias); }
    case 4u: { return textureSampleBias(tex4, tex4_samp, uv, bias); }
    case 5u: { return textureSampleBias(tex5, tex5_samp, uv, bias); }
    case 6u: { return textureSampleBias(tex6, tex6_samp, uv, bias); }
    default: { return textureSampleBias(tex7, tex7_samp, uv, bias); }
  }
}

float wrap_component(uint wrap, float coord) {
  if (wrap == 0u) { return coord; }
  if (wrap >= 6u) { return 0.0; }
  return coord % float(512u >> wrap);
}

static float4 regs[4];
static float4 rast[2];
static float3 ind_coord[4];
static float4 stage_sample;
static uint stage_konst;
static uint stage_tex;
static uint stage_ind;

float4 swizzled(float4 value, uint swizzle) {
  return float4(value[swizzle & 3u], value[(swizzle >> 2u) & 3u], value[(swizzle >> 4u) & 3u], value[(swizzle >> 6u) & 3u]);
}

float4 stage_texture() {
  return swizzled(stage_sample, byte_of(stage_tex, 3u));
}

// What the stage rasterizes; an alpha bump fills all four components.
float4 stage_rasterized() {
  uint kind = byte_of(stage_konst, 2u);
  if (kind == 0u) {
    return vec4f(0.0);
  }
  if (kind <= 2u) {
    return swizzled(rast[kind - 1u], byte_of(stage_tex, 2u));
  }
  float coord = ind_coord[nibble_of(stage_ind, 0u)][nibble_of(stage_ind, 3u) - 1u];
  float alpha = float(uint(coord) & byte_of(stage_konst, 3u)) / 255.0;
  if (kind == 4u) {
    alpha = alpha * (255.0 / 248.0);
  }
  return vec4f(alpha);
}

float konst_fraction(uint selection) {
  return float(8u - selection) / 8.0;
}

float konst_component(uint selection) {
  uint from_k0 = selection - 0x10u;
  return ubuf.kcolors[from_k0 & 3u][from_k0 >> 2u];
}

float3 color_arg(uint arg) {
  switch (arg) {
    case 0u: { return regs[0].rgb; }
    case 1u: { return vec3f(regs[0].a); }
    case 2u: { return regs[1].rgb; }
    case 3u: { return vec3f(regs[1].a); }
    case 4u: { return regs[2].rgb; }
    case 5u: { return vec3f(regs[2].a); }
    case 6u: { return regs[3].rgb; }
    case 7u: { return vec3f(regs[3].a); }
    case 8u: { return stage_texture().rgb; }
    case 9u: { return vec3f(stage_texture().a); }
    case 10u: { return stage_rasterized().rgb; }
    case 11u: { return vec3f(stage_rasterized().a); }
    case 12u: { return vec3f(1.0); }
    case 13u: { return vec3f(0.5); }
    case 14u: {
      uint selection = byte_of(stage_konst, 0u);
      if (selection < 8u) { return vec3f(konst_fraction(selection)); }
      if (selection < 0x10u) { return ubuf.kcolors[selection - 0xCu].rgb; }
      return vec3f(konst_component(selection));
    }
    default: { return vec3f(0.0); }
  }
}

float alpha_arg(uint arg) {
  switch (arg) {
    case 0u: { return regs[0].a; }
    case 1u: { return regs[1].a; }
    case 2u: { return regs[2].a; }
    case 3u: { return regs[3].a; }
    case 4u: { return stage_texture().a; }
    case 5u: { return stage_rasterized().a; }
    case 6u: {
      uint selection = byte_of(stage_konst, 1u);
      if (selection < 8u) { return konst_fraction(selection); }
      return konst_component(selection);
    }
    default: { return 0.0; }
  }
}

float tev_bias(uint bias) {
  if (bias == 1u) { return 0.5; }
  if (bias == 2u) { return -0.5; }
  return 0.0;
}

float tev_scale(uint scale) {
  switch (scale) {
    case 1u: { return 2.0; }
    case 2u: { return 4.0; }
    case 3u: { return 0.5; }
    default: { return 1.0; }
  }
}

float rg16(float3 v) {
  return round(dot(v.rg * 255.0, vec2(1.0, 256.0)));
}
float bgr24(float3 v) {
  return round(dot(v.rgb * 255.0, vec3(1.0, 256.0, 65536.0)));
}
float r8(float3 v) {
  return round(v.r * 255.0);
}

// The packed comparisons: whether a's value passes against b's.
bool packed_compare(uint op, float3 a, float3 b) {
  switch (op) {
    case 8u: { return r8(a) > r8(b); }
    case 9u: { return r8(a) == r8(b); }
    case 10u: { return rg16(a) > rg16(b); }
    case 11u: { return rg16(a) == rg16(b); }
    case 12u: { return bgr24(a) > bgr24(b); }
    default: { return bgr24(a) == bgr24(b); }
  }
}

float3 tev_color(uint op_word, bool clamped, float3 a, float3 b, float3 c, float3 d) {
  uint op = byte_of(op_word, 0u);
  float3 result;
  if (op <= 1u) {
    float3 mixed = select(mix(a, b, c), -mix(a, b, c), op == 1u);
    result = (mixed + d + tev_bias(byte_of(op_word, 1u))) * tev_scale(byte_of(op_word, 2u));
  } else if (op <= 13u) {
    result = select(vec3(0.0), c, packed_compare(op, a, b)) + d;
  } else if (op == 14u) {
    result = select(vec3(0.0), c, round(a * 255.0) > round(b * 255.0)) + d;
  } else {
    result = select(vec3(0.0), c, round(a * 255.0) == round(b * 255.0)) + d;
  }
  if (clamped) {
    return clamp(result, vec3f(0.0), vec3f(1.0));
  }
  return clamp(result, vec3f(-4.0), vec3f(4.0));
}

float tev_alpha(uint op_word, bool clamped, float a, float b, float c, float d, float3 color_a, float3 color_b) {
  uint op = byte_of(op_word, 0u);
  float result;
  if (op <= 1u) {
    float mixed = select(mix(a, b, c), -mix(a, b, c), op == 1u);
    result = (mixed + d + tev_bias(byte_of(op_word, 1u))) * tev_scale(byte_of(op_word, 2u));
  } else if (op <= 13u) {
    result = select(0.0, c, packed_compare(op, color_a, color_b)) + d;
  } else if (op == 14u) {
    result = select(0.0, c, round(a * 255.0) > round(b * 255.0)) + d;
  } else {
    result = select(0.0, c, round(a * 255.0) == round(b * 255.0)) + d;
  }
  if (clamped) {
    return clamp(result, 0.0, 1.0);
  }
  return clamp(result, -4.0, 4.0);
}

bool compares(uint comp, uint value, uint reference) {
  switch (comp) {
    case 0u: { return false; }
    case 1u: { return value < reference; }
    case 2u: { return value == reference; }
    case 3u: { return value <= reference; }
    case 4u: { return value > reference; }
    case 5u: { return value != reference; }
    case 6u: { return value >= reference; }
    default: { return true; }
  }
}

float3 tex_coord(VertexOutput input, uint i) {
  switch (i) {
    case 0u: { return input.tex0; }
    case 1u: { return input.tex1; }
    case 2u: { return input.tex2; }
    case 3u: { return input.tex3; }
    case 4u: { return input.tex4; }
    case 5u: { return input.tex5; }
    case 6u: { return input.tex6; }
    default: { return input.tex7; }
  }
}

float4 tev_shade(VertexOutput input) {
    uint loads = cfg(CFG_LOADS_TEV_REG);
    for (uint i = 0u; i < 4u; i++) {
        if ((loads & (1u << i)) != 0u) {
            regs[i] = ubuf.tev_regs[i];
        }
    }
    rast[0] = input.cc0;
    rast[1] = input.cc1;

    float2 tex_uv[8];
    for (uint i = 0u; i < 8u; i++) {
        tex_uv[i] = float2(0.0, 0.0);
    }
    uint sampled_coords = cfg(CFG_SAMPLED_COORDS);
    for (uint i = 0u; i < 8u; i++) {
        if ((sampled_coords & (1u << i)) == 0u) { continue; }
        float3 coord = tex_coord(input, i);
        if (byte_of(cfg(CFG_TCGS + i * 2u), 0u) == TG_MTX3X4) {
            tex_uv[i] = coord.xy / coord.z;
        } else {
            tex_uv[i] = coord.xy;
        }
    }

    uint used_ind = cfg(CFG_USED_IND_STAGES);
    for (uint i = 0u; i < 4u; i++) {
        if ((used_ind & (1u << i)) == 0u) { continue; }
        uint w0 = cfg(CFG_IND + i * 2u);
        uint w1 = cfg(CFG_IND + i * 2u + 1u);
        uint coord = byte_of(w0, 0u);
        uint map = byte_of(w0, 1u);
        float2 scale = vec2f(1.0 / float(1u << byte_of(w1, 0u)), 1.0 / float(1u << byte_of(w1, 1u)));
        float2 uv = tex_uv[coord] * ubuf.texcoord_scale[coord].xy * scale / ubuf.tex_size_bias[map].xy;
        ind_coord[i] = 255.0 * sample_map(map, uv, ubuf.tex_size_bias[map].z).abg;
    }

    // Each stage's texture, with the texcoords the indirect stages carry on.
    float4 samples[16];
    for (uint i = 0u; i < 16u; i++) {
        samples[i] = float4(0.0, 0.0, 0.0, 0.0);
    }
    float2 t_tex_coord = vec2f(0.0);
    uint stages = cfg(CFG_TEV_STAGES);
    for (uint i = 0u; i < stages; i++) {
        uint base = CFG_STAGES + i * CFG_STAGE_WORDS;
        uint flags = cfg(base + 7u);
        if ((flags & (FLAG_NEEDS_TEV_COORD | FLAG_NEEDS_SAMPLE)) == 0u) { continue; }
        uint tex = cfg(base + 5u);
        uint coord = byte_of(tex, 0u);
        uint map = byte_of(tex, 1u);
        bool uv_set = false;
        float2 uv = float2(0.0, 0.0);
        if ((flags & FLAG_NEEDS_TEV_COORD) != 0u) {
            uint ind = cfg(base + 6u);
            uint matrix_kind = nibble_of(ind, 4u);
            float2 offset = vec2f(0.0);
            if (matrix_kind != 0u) {
                uint format = nibble_of(ind, 1u);
                float3 ind_value = ind_coord[nibble_of(ind, 0u)];
                if (format != 0u) {
                    ind_value = floor(ind_value / float(4u << format));
                }
                uint bias_sel = nibble_of(ind, 2u);
                if (bias_sel != 0u) {
                    float bias = select(1.0, -128.0, format == 0u);
                    ind_value = ind_value + vec3f(float(bias_sel & 1u), float((bias_sel >> 1u) & 1u), float((bias_sel >> 2u) & 1u)) * bias;
                }
                uint matrix_index = nibble_of(ind, 5u);
                float4 c0 = ubuf.ind_mtx[matrix_index * 2u];
                float4 c1 = ubuf.ind_mtx[matrix_index * 2u + 1u];
                if (matrix_kind == 1u) {
                    offset = vec2f(dot(vec3f(c0.xz, c1.x), ind_value), dot(vec3f(c0.yw, c1.y), ind_value)) * c1.z;
                } else if (matrix_kind == 2u) {
                    offset = tex_uv[coord] * ubuf.texcoord_scale[coord].xy * ind_value.x * c1.z / 256.0;
                } else {
                    offset = tex_uv[coord] * ubuf.texcoord_scale[coord].xy * ind_value.y * c1.z / 256.0;
                }
            }
            if ((flags & FLAG_WRITES_COORD) != 0u) {
                float2 final_coord = offset;
                if ((flags & FLAG_HAS_BASE_COORD) != 0u) {
                    float2 base_coord = tex_uv[coord];
                    if ((flags & FLAG_SIMPLE_COORDS) == 0u) {
                        base_coord = base_coord * ubuf.texcoord_scale[coord].xy;
                    }
                    float2 wrapped = vec2f(wrap_component(nibble_of(ind, 6u), base_coord.x), wrap_component(nibble_of(ind, 7u), base_coord.y));
                    final_coord = wrapped + offset;
                }
                if ((flags & FLAG_ADD_PREVIOUS) != 0u) {
                    t_tex_coord += final_coord;
                } else {
                    t_tex_coord = final_coord;
                }
                if ((flags & FLAG_NEEDS_SAMPLE) != 0u && (flags & FLAG_HAS_BASE_TEXTURE) != 0u) {
                    uv = t_tex_coord;
                    if ((flags & FLAG_SIMPLE_COORDS) == 0u) {
                        uv = uv / ubuf.tex_size_bias[map].xy;
                    }
                    uv_set = true;
                }
            }
        }
        if ((flags & FLAG_NEEDS_SAMPLE) == 0u) { continue; }
        if ((flags & FLAG_HAS_BASE_TEXTURE) == 0u) {
            // Tex order disabled: the hardware feeds TEV white.
            samples[i] = vec4f(1.0);
            continue;
        }
        if (!uv_set) {
            uv = tex_uv[coord];
        }
        samples[i] = sample_map(map, uv, ubuf.tex_size_bias[map].z);
    }

    for (uint i = 0u; i < stages; i++) {
        uint base = CFG_STAGES + i * CFG_STAGE_WORDS;
        uint color_args = cfg(base);
        uint alpha_args = cfg(base + 1u);
        uint color_op = cfg(base + 2u);
        uint alpha_op = cfg(base + 3u);
        uint flags = cfg(base + 7u);
        stage_konst = cfg(base + 4u);
        stage_tex = cfg(base + 5u);
        stage_ind = cfg(base + 6u);
        stage_sample = samples[i];

        float3 color_a = color_arg(byte_of(color_args, 0u));
        float3 color_b = color_arg(byte_of(color_args, 1u));
        float3 color_c = color_arg(byte_of(color_args, 2u));
        if ((flags & FLAG_COLOR_OVERFLOW_A) != 0u) { color_a = tev_overflow_vec3f(color_a); }
        if ((flags & (FLAG_COLOR_OVERFLOW_A << 1u)) != 0u) { color_b = tev_overflow_vec3f(color_b); }
        if ((flags & (FLAG_COLOR_OVERFLOW_A << 2u)) != 0u) { color_c = tev_overflow_vec3f(color_c); }
        float3 color_result = tev_color(color_op, (flags & FLAG_COLOR_CLAMP) != 0u, color_a, color_b, color_c, color_arg(byte_of(color_args, 3u)));
        uint color_out = byte_of(color_op, 3u);
        regs[color_out] = vec4f(color_result, regs[color_out].a);

        float alpha_a = alpha_arg(byte_of(alpha_args, 0u));
        float alpha_b = alpha_arg(byte_of(alpha_args, 1u));
        float alpha_c = alpha_arg(byte_of(alpha_args, 2u));
        if ((flags & FLAG_ALPHA_OVERFLOW_A) != 0u) { alpha_a = tev_overflow_f32(alpha_a); }
        if ((flags & (FLAG_ALPHA_OVERFLOW_A << 1u)) != 0u) { alpha_b = tev_overflow_f32(alpha_b); }
        if ((flags & (FLAG_ALPHA_OVERFLOW_A << 2u)) != 0u) { alpha_c = tev_overflow_f32(alpha_c); }
        uint alpha_out = byte_of(alpha_op, 3u);
        regs[alpha_out].a = tev_alpha(alpha_op, (flags & FLAG_ALPHA_CLAMP) != 0u, alpha_a, alpha_b, alpha_c, alpha_arg(byte_of(alpha_args, 3u)), color_a, color_b);
    }

    float4 prev = regs[0];
    uint last_outputs = cfg(CFG_LAST_OUTPUTS);
    if (byte_of(last_outputs, 0u) != 0u) {
        prev = vec4f(regs[byte_of(last_outputs, 0u)].rgb, prev.a);
    }
    if (byte_of(last_outputs, 1u) != 0u) {
        prev.a = regs[byte_of(last_outputs, 1u)].a;
    }

    uint fog_type = cfg(CFG_FOG_TYPE);
    if (fog_type != 0u) {
        float fog_depth = 1.0 - input.pos.z;
        float fog_base;
        if ((fog_type & 8u) != 0u) {
            fog_base = ubuf.fog.a * fog_depth;
        } else {
            fog_base = ubuf.fog.a / (ubuf.fog.b - fog_depth);
        }
        if (cfg(CFG_FOG_RANGE) != 0u) {
            fog_base *= asfloat(abuf.Load((ubuf.fog_range_base + uint(input.pos.x)) * 4u));
        }
        float fog_f = clamp(fog_base - ubuf.fog.c, 0.0, 1.0);
        float fog_z;
        switch (fog_type & 7u) {
            case 2u: { fog_z = fog_f; break; }
            case 4u: { fog_z = 1.0 - exp2(-8.0 * fog_f); break; }
            case 5u: { fog_z = 1.0 - exp2(-8.0 * fog_f * fog_f); break; }
            case 6u: { fog_z = exp2(-8.0 * (1.0 - fog_f)); break; }
            default: {
                fog_f = 1.0 - fog_f;
                fog_z = exp2(-8.0 * fog_f * fog_f);
                break;
            }
        }
        prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb, clamp(fog_z, 0.0, 1.0)), prev.a);
    }

    uint prev_overflow = cfg(CFG_PREV_OVERFLOW);
    if ((prev_overflow & 1u) != 0u) {
        prev = vec4f(tev_overflow_vec3f(prev.rgb), prev.a);
    }
    if ((prev_overflow & 2u) != 0u) {
        prev.a = tev_overflow_f32(prev.a);
    }

    float alpha_width = max(fwidth(prev.a), 0.0001);
    if (cfg(CFG_ALPHA_COMPARE) != 0u) {
        if (cfg(CFG_ALPHA_TO_COVERAGE) != 0u) {
            // Coverage from how far alpha is past the edge, in pixels.
            prev.a = clamp((prev.a - cfg_f32(CFG_COVERAGE_THRESHOLD)) / alpha_width + 0.5, 0.0, 1.0);
        } else {
            uint alpha = uint(round(clamp(prev.a, 0.0, 1.0) * 255.0));
            bool pass0 = compares(cfg(CFG_ALPHA_COMP0), alpha, cfg(CFG_ALPHA_REF0));
            bool pass1 = compares(cfg(CFG_ALPHA_COMP1), alpha, cfg(CFG_ALPHA_REF1));
            bool passes;
            switch (cfg(CFG_ALPHA_OP)) {
                case 0u: { passes = pass0 && pass1; break; }
                case 1u: { passes = pass0 || pass1; break; }
                case 2u: { passes = pass0 != pass1; break; }
                default: { passes = pass0 == pass1; break; }
            }
            if (!passes) { discard; }
        }
    }
    return prev;
}
)""";
}
} // namespace aurora::gx::hlsl
