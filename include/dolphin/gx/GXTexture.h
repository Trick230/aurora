#ifndef DOLPHIN_GXTEXTURE_H
#define DOLPHIN_GXTEXTURE_H

#include <dolphin/gx/GXEnum.h>
#include <dolphin/gx/GXStruct.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef GXTexRegion* (*GXTexRegionCallback)(const GXTexObj* obj, GXTexMapID id);

u32 GXGetTexBufferSize(u16 width, u16 height, u32 format, GXBool mipmap, u8 max_lod);

#ifdef __cplusplus
}
#endif

#endif
