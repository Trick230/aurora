#pragma once

#include <cstdint>
#include <vector>

struct GXTexObj_;
struct GXTlutObj_;

namespace aurora::gx::recomp {

enum class AddressSpace : std::uint32_t {
  Auto = 0,
  Virtual,
  Physical,
};

enum class ResourceKind : std::uint32_t {
  Generic = 0,
  Fifo,
  DisplayList,
  VertexArray,
  Texture,
  Tlut,
  CopyDestination,
};

using ResolveGuestAddressFn = bool (*)(void* user, std::uint32_t address, std::uint32_t size,
                                       AddressSpace space, ResourceKind resource, const void** data,
                                       std::uint32_t* available);

void set_guest_address_resolver(ResolveGuestAddressFn resolve, void* user);
void clear_guest_address_resolver();
bool resolve_guest_address(std::uint32_t address, std::uint32_t size, AddressSpace space, ResourceKind resource,
                           const void** data, std::uint32_t* available);

// A retail FIFO packet the host has to act on, reported by process_stream.
enum class StreamEvent : std::uint32_t {
  None = 0,
  DrawDone,     // BP PE_DONE (GXSetDrawDone)
  Token,        // BP PE_TOKEN (GXSetDrawSync without interrupt), value = token
  TokenInt,     // BP PE_TOKEN_INT, value = token
  DisplayCopy,  // BP copy trigger to the XFB (GXCopyDisp); the frame is done
};

// Besides retail packets the stream may carry this host one, which hooks in a
// recompiled game write through the write-gather pipe like any other: the
// opcode, then a view matrix as 12 big-endian floats (row-major 3x4, the world
// to eye space). The draws after it, up to the next one or the frame's end,
// had their matrices built with that camera; frame interpolation then blends
// their objects in the world and follows the camera's own move. No retail
// opcode has these bits.
constexpr std::uint8_t HostViewOpcode = 0x70;
constexpr std::uint32_t HostViewSize = 1 + 12 * 4;

// And GX calls the host makes in place of the game's code (GX at the API): the
// opcode, the call, its index, then its data as the game keeps it (big-endian
// words). The host writes them where the call would have written its FIFO
// packets, so they keep their order with the vertices and everything else;
// the call's XF address and size are worked out here.
constexpr std::uint8_t HostGxOpcode = 0x78;
enum class HostGxCall : std::uint8_t {
  LoadPosMtx,   // GXLoadPosMtxImm: index the matrix id, a 3x4 matrix
  LoadNrmMtx,   // GXLoadNrmMtxImm: index the matrix id, a 3x3 matrix (rows of 3)
  LoadTexMtx34, // GXLoadTexMtxImm with GX_MTX3x4: index the GXTexMtx id, a 3x4 matrix
  LoadTexMtx24, // GXLoadTexMtxImm with GX_MTX2x4: index the GXTexMtx id, a 2x4 matrix
  LoadLight,    // GXLoadLightObjImm: index the light (0-7), the GXLightObj from its color on (13 words)
  Count,
};
constexpr std::uint32_t HostGxHeaderSize = 3;
// The packet's size from its call byte, 0 for an unknown call.
constexpr std::uint32_t host_gx_size(std::uint8_t call) {
  constexpr std::uint8_t words[] = {12, 9, 12, 8, 13};
  return call < static_cast<std::uint8_t>(HostGxCall::Count) ? HostGxHeaderSize + words[call] * 4u : 0;
}

// Processes the complete packets at the front of a big-endian retail FIFO
// stream, as the hardware command processor would, and returns how many bytes
// it consumed; whatever is left is the start of a packet still being written.
// It stops right after a packet that raises a StreamEvent so the host can
// signal the guest in order.
std::uint32_t process_stream(const std::uint8_t* data, std::uint32_t size, StreamEvent* event,
                             std::uint32_t* value);

// The last value written to every GX register, as big-endian FIFO packets that
// bring the decoder back to that state when processed (a savestate restore).
// Call only while no stream is being processed.
std::vector<std::uint8_t> register_stream();

std::uint32_t texture_byte_size(const GXTexObj_& obj) noexcept;
bool resolve_texture_data(GXTexObj_& obj) noexcept;
bool resolve_texture_tlut(std::uint32_t textureSlot, GXTlutObj_* out) noexcept;

} // namespace aurora::gx::recomp
