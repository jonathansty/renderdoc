/******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Baldur Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ******************************************************************************/

#pragma once

#include "vk_info.h"

// Resolver for VkShaderDescriptorSetAndBindingMappingInfoEXT: given a set/binding/resource type
// (and array index) in a shader on a descriptor heap pipeline or shader object, work out where the
// descriptor (or data) lives. This is the single implementation of the spec address formulas, so
// that static access, feedback decoding and the shader debugger all agree.

enum class HeapMappingKind
{
  // no mapping matched, or the result slot is unused (e.g. no sampler part)
  None,
  // byteOffset is a byte offset of a descriptor in the resource heap
  ResourceHeap,
  // byteOffset is a byte offset of a descriptor in the sampler heap
  SamplerHeap,
  // the sampler is the one embedded in the mapping (DescriptorMapping::sampler)
  EmbeddedSampler,
  // a uniform buffer backed by push data, byteOffset is the offset into push data
  PushData,
  // buffer/AS backed by an address in push data, byteOffset is the offset into push data of the
  // address and 'address' is the address itself if push data was available
  PushAddress,
  // a uniform buffer backed directly by resource heap *data* (not a descriptor), byteOffset is the
  // offset into the resource heap
  HeapData,
  // the mapping was found but resolving it needs memory reads (indirect index/array/address, shader
  // record) or push data that wasn't provided/was out of range
  Unresolved,
};

struct HeapMappingResult
{
  HeapMappingKind kind = HeapMappingKind::None;
  // index into the descriptorMappings array that matched
  uint32_t mappingIndex = ~0U;
  // byte offset into the heap (or push data offset for PushData/PushAddress)
  uint64_t byteOffset = 0;
  // the device address read from push data for PushAddress
  uint64_t address = 0;

  // for combined image samplers, the sampler part (SamplerHeap, EmbeddedSampler or Unresolved)
  HeapMappingKind samplerKind = HeapMappingKind::None;
  uint64_t samplerByteOffset = 0;
};

// the VkSpirvResourceTypeFlagBitsEXT mask a reflected binding corresponds to. Reflection does not
// distinguish NonWritable storage buffers/images so read-write bindings return both the read-only
// and read-write bits (a mapping matches if it overlaps any bit). Returns 0 for things which can't
// be mapped (push constants, spec constants, unknown).
VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForDescriptor(DescriptorType type,
                                                           bool isInputAttachment = false);
VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForBinding(const ConstantBlock &bind);
VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForBinding(const ShaderSampler &bind);
VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForBinding(const ShaderResource &bind);

// find the first mapping covering set/binding with a resourceMask overlapping 'type'. Returns NULL
// and sets *mappingIndex to ~0U if there is none.
const VulkanCreationInfo::ShaderEntry::DescriptorMapping *FindHeapMapping(
    const rdcarray<VulkanCreationInfo::ShaderEntry::DescriptorMapping> &mappings, uint32_t set,
    uint32_t binding, VkSpirvResourceTypeFlagsEXT type, uint32_t *mappingIndex = NULL);

// find and resolve the mapping for a binding. pushData can be NULL if not available, in which case
// push-data-dependent sources return Unresolved.
HeapMappingResult ResolveHeapMapping(
    const rdcarray<VulkanCreationInfo::ShaderEntry::DescriptorMapping> &mappings, uint32_t set,
    uint32_t binding, VkSpirvResourceTypeFlagsEXT type, uint32_t arrayIndex, const byte *pushData,
    size_t pushDataSize);
