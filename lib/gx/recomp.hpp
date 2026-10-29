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
