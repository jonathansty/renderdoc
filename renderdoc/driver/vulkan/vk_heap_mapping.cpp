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

#include "vk_heap_mapping.h"

VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForDescriptor(DescriptorType type, bool isInputAttachment)
{
  switch(type)
  {
    case DescriptorType::ConstantBuffer: return VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    case DescriptorType::Sampler: return VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    case DescriptorType::ImageSampler: return VK_SPIRV_RESOURCE_TYPE_COMBINED_SAMPLED_IMAGE_BIT_EXT;
    case DescriptorType::Image:
      // input attachments are Sampled=2 subpass data images, which are never writable
      return isInputAttachment ? VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT
                               : VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    // uniform texel buffers are Sampled=1 images
    case DescriptorType::TypedBuffer: return VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    case DescriptorType::ReadWriteImage:
    case DescriptorType::ReadWriteTypedBuffer:
      return VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
             VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT;
    case DescriptorType::Buffer:
    case DescriptorType::ReadWriteBuffer:
      return VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
             VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    case DescriptorType::AccelerationStructure:
      return VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    default: break;
  }

  return 0;
}

VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForBinding(const ConstantBlock &bind)
{
  // push constants and spec constants are not mappable
  if(!bind.bufferBacked)
    return 0;

  return VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
}

VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForBinding(const ShaderSampler &bind)
{
  return VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
}

VkSpirvResourceTypeFlagsEXT SpirvResourceTypeForBinding(const ShaderResource &bind)
{
  return SpirvResourceTypeForDescriptor(bind.descriptorType, bind.isInputAttachment);
}

const VulkanCreationInfo::ShaderEntry::DescriptorMapping *FindHeapMapping(
    const rdcarray<VulkanCreationInfo::ShaderEntry::DescriptorMapping> &mappings, uint32_t set,
    uint32_t binding, VkSpirvResourceTypeFlagsEXT type, uint32_t *mappingIndex)
{
  if(mappingIndex)
    *mappingIndex = ~0U;

  if(type == 0)
    return NULL;

  // the spec requires that no two mappings overlap in set, binding range and resource mask, so
  // there is at most one match for a single resource type. For reflected types where the type mask
  // has multiple bits (see SpirvResourceTypeForDescriptor) the first overlapping mapping wins.
  for(size_t i = 0; i < mappings.size(); i++)
  {
    const VkDescriptorSetAndBindingMappingEXT &m = mappings[i].mapping;

    if(m.descriptorSet != set)
      continue;
    if(((uint32_t)m.resourceMask & (uint32_t)type) == 0)
      continue;
    // compare in 64-bit so first+count can't wrap
    if(binding < m.firstBinding || (uint64_t)binding >= (uint64_t)m.firstBinding + m.bindingCount)
      continue;

    if(mappingIndex)
      *mappingIndex = (uint32_t)i;
    return &mappings[i];
  }

  return NULL;
}

static bool ReadPushU32(const byte *pushData, size_t pushDataSize, uint64_t offset, uint32_t &value)
{
  if(!pushData || offset + sizeof(uint32_t) > pushDataSize)
    return false;

  memcpy(&value, pushData + offset, sizeof(uint32_t));
  return true;
}

static bool ReadPushU64(const byte *pushData, size_t pushDataSize, uint64_t offset, uint64_t &value)
{
  if(!pushData || offset + sizeof(uint64_t) > pushDataSize)
    return false;

  memcpy(&value, pushData + offset, sizeof(uint64_t));
  return true;
}

// reads an address from push data then a value from memory at that address plus an offset
template <typename T>
static bool ReadIndirect(const byte *pushData, size_t pushDataSize,
                         const HeapMemoryReader &readMemory, uint64_t pushOffset,
                         uint64_t addressOffset, T &value)
{
  uint64_t indirectAddress = 0;
  if(!readMemory || !ReadPushU64(pushData, pushDataSize, pushOffset & ~7ULL, indirectAddress))
    return false;

  return readMemory(indirectAddress + addressOffset, &value, sizeof(T));
}

/*
 * Spec formulas (VK_EXT_descriptor_heap, "Shader Bindings"), all offsets are byte offsets:
 *
 * A mapping applies to resources with DescriptorSet == descriptorSet, a Binding in
 * [firstBinding, firstBinding + bindingCount) and a type matching a bit in resourceMask.
 * Every source below with a heap index uses:
 *   shaderIndex = (Binding - firstBinding) + arrayIndex
 *
 * HEAP_WITH_CONSTANT_OFFSET:
 *   offset = heapOffset + (shaderIndex * heapArrayStride)
 *   "If the mapped resource is a OpTypeSampledImage, offset is instead calculated for the sampler as
 *    offset = samplerHeapOffset + (shaderIndex * samplerHeapArrayStride)"
 *
 * HEAP_WITH_PUSH_INDEX:
 *   pushIndex = ((uint32_t*)pPushData)[pushOffset/4]
 *   offset = heapOffset + (pushIndex * heapIndexStride) + (shaderIndex * heapArrayStride)
 *   sampler for OpTypeSampledImage:
 *   samplerPushIndex = ((uint32_t*)pPushData)[samplerPushOffset/4]
 *   offset = samplerHeapOffset + (samplerPushIndex * samplerHeapIndexStride) +
 *            (shaderIndex * samplerHeapArrayStride)
 *   "If useCombinedImageSamplerIndex is VK_TRUE, and the mapped resource is a OpTypeSampledImage,
 *    pushIndex and samplerPushIndex in the above equations are instead calculated as
 *    pushIndex = ((uint32_t*)pPushData)[pushOffset/4] & 0xFFFFF
 *    samplerPushIndex = (((uint32_t*)pPushData)[pushOffset/4] >> 20) & 0xFFF"
 *
 * HEAP_WITH_INDIRECT_INDEX:
 *   indirectAddress = ((VkDeviceAddress*)pPushData)[pushOffset/8]
 *   indirectIndex = ((uint32_t*)indirectAddress)[addressOffset/4]
 *   offset = heapOffset + (indirectIndex * heapIndexStride) + (shaderIndex * heapArrayStride)
 *   (sampler/combined index variants as for push index, with the sampler* members)
 *
 * HEAP_WITH_INDIRECT_INDEX_ARRAY:
 *   indirectAddress = ((VkDeviceAddress*)pPushData)[pushOffset/8]
 *   indirectIndex = ((uint32_t*)indirectAddress)[(addressOffset/4) + shaderIndex]
 *   offset = heapOffset + (indirectIndex * heapIndexStride)     (no array stride)
 *
 * RESOURCE_HEAP_DATA (uniform buffer backed by heap data, not a descriptor):
 *   offset = shaderOffset + heapOffset + ((uint32_t*)pPushData)[pushOffset/4]
 *
 * PUSH_DATA (uniform buffer backed by push data):
 *   offset = shaderOffset + pushDataOffset
 *
 * PUSH_ADDRESS:
 *   address = ((VkDeviceAddress*)pPushData)[pushAddressOffset/8]
 *
 * INDIRECT_ADDRESS:
 *   indirectAddress = ((VkDeviceAddress*)pPushData)[pushOffset/8]
 *   resourceAddress = ((VkDeviceAddress*)indirectAddress)[addressOffset/8]
 *
 * HEAP_WITH_SHADER_RECORD_INDEX / SHADER_RECORD_DATA / SHADER_RECORD_ADDRESS: as the push
 * equivalents but sourced from the shader record.
 *
 * Embedded samplers: "If the mapped resource is a OpTypeSampler or OpTypeSampledImage, and
 * pEmbeddedSampler is not NULL, the specified embedded sampler will be used rather than accessing
 * the sampler heap." Only OpTypeSampledImage looks in the sampler heap with the sampler* members; a
 * plain OpTypeSampler uses the main heapOffset/heapArrayStride/etc members, but in the sampler heap.
 */
HeapMappingResult ResolveHeapMapping(
    const rdcarray<VulkanCreationInfo::ShaderEntry::DescriptorMapping> &mappings, uint32_t set,
    uint32_t binding, VkSpirvResourceTypeFlagsEXT type, uint32_t arrayIndex, const byte *pushData,
    size_t pushDataSize, const HeapMemoryReader &readMemory)
{
  HeapMappingResult ret;

  uint32_t mappingIndex = ~0U;
  const VulkanCreationInfo::ShaderEntry::DescriptorMapping *dm =
      FindHeapMapping(mappings, set, binding, type, &mappingIndex);

  if(!dm)
    return ret;

  ret.mappingIndex = mappingIndex;

  const VkDescriptorSetAndBindingMappingEXT &m = dm->mapping;
  const VkDescriptorMappingSourceDataEXT &src = m.sourceData;

  const bool isSampler = (type & VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT) != 0;
  const bool isCombined = (type & VK_SPIRV_RESOURCE_TYPE_COMBINED_SAMPLED_IMAGE_BIT_EXT) != 0;

  // the heap the main (non-sampler-part) descriptor lives in
  const HeapMappingKind mainHeap =
      isSampler ? HeapMappingKind::SamplerHeap : HeapMappingKind::ResourceHeap;

  const uint64_t shaderIndex = uint64_t(binding - m.firstBinding) + arrayIndex;

  // pEmbeddedSampler, if the source can have one. The stored copy has the pointer NULL'd out and
  // hasEmbeddedSampler set instead
  const bool embedded = dm->hasEmbeddedSampler && (isSampler || isCombined);

  switch(m.source)
  {
    case VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT:
    {
      const VkDescriptorMappingSourceConstantOffsetEXT &c = src.constantOffset;

      if(embedded && isSampler)
      {
        ret.kind = HeapMappingKind::EmbeddedSampler;
        break;
      }

      ret.kind = mainHeap;
      ret.byteOffset = uint64_t(c.heapOffset) + shaderIndex * c.heapArrayStride;

      if(isCombined)
      {
        if(embedded)
        {
          ret.samplerKind = HeapMappingKind::EmbeddedSampler;
        }
        else
        {
          ret.samplerKind = HeapMappingKind::SamplerHeap;
          ret.samplerByteOffset =
              uint64_t(c.samplerHeapOffset) + shaderIndex * c.samplerHeapArrayStride;
        }
      }
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT:
    {
      const VkDescriptorMappingSourcePushIndexEXT &p = src.pushIndex;

      if(embedded && isSampler)
      {
        ret.kind = HeapMappingKind::EmbeddedSampler;
        break;
      }

      const bool packed = isCombined && p.useCombinedImageSamplerIndex;

      uint32_t pushIndex = 0;
      const bool ok = ReadPushU32(pushData, pushDataSize, p.pushOffset & ~3U, pushIndex);

      uint32_t samplerPushIndex = 0;
      bool samplerOk = ok;

      if(packed)
      {
        samplerPushIndex = (pushIndex >> 20) & 0xFFF;
        pushIndex &= 0xFFFFF;
      }
      else if(isCombined && !embedded)
      {
        samplerOk = ReadPushU32(pushData, pushDataSize, p.samplerPushOffset & ~3U, samplerPushIndex);
      }

      if(ok)
      {
        ret.kind = mainHeap;
        ret.byteOffset = uint64_t(p.heapOffset) + uint64_t(pushIndex) * p.heapIndexStride +
                         shaderIndex * p.heapArrayStride;
      }
      else
      {
        ret.kind = HeapMappingKind::Unresolved;
      }

      if(isCombined)
      {
        if(embedded)
        {
          ret.samplerKind = HeapMappingKind::EmbeddedSampler;
        }
        else if(samplerOk)
        {
          ret.samplerKind = HeapMappingKind::SamplerHeap;
          ret.samplerByteOffset = uint64_t(p.samplerHeapOffset) +
                                  uint64_t(samplerPushIndex) * p.samplerHeapIndexStride +
                                  shaderIndex * p.samplerHeapArrayStride;
        }
        else
        {
          ret.samplerKind = HeapMappingKind::Unresolved;
        }
      }
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT:
    case VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT:
    {
      // the two sources only differ in how the array index is used: the array form reads one index
      // per array element and has no array stride
      const bool isArray =
          (m.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT);

      uint32_t heapOffset, pushOffset, addressOffset, indexStride, arrayStride;
      bool useCombined;
      uint32_t samplerHeapOffset, samplerPushOffset, samplerAddressOffset, samplerIndexStride,
          samplerArrayStride;

      if(isArray)
      {
        const VkDescriptorMappingSourceIndirectIndexArrayEXT &a = src.indirectIndexArray;
        heapOffset = a.heapOffset;
        pushOffset = a.pushOffset;
        addressOffset = a.addressOffset;
        indexStride = a.heapIndexStride;
        arrayStride = 0;
        useCombined = a.useCombinedImageSamplerIndex != VK_FALSE;
        samplerHeapOffset = a.samplerHeapOffset;
        samplerPushOffset = a.samplerPushOffset;
        samplerAddressOffset = a.samplerAddressOffset;
        samplerIndexStride = a.samplerHeapIndexStride;
        samplerArrayStride = 0;
      }
      else
      {
        const VkDescriptorMappingSourceIndirectIndexEXT &a = src.indirectIndex;
        heapOffset = a.heapOffset;
        pushOffset = a.pushOffset;
        addressOffset = a.addressOffset;
        indexStride = a.heapIndexStride;
        arrayStride = a.heapArrayStride;
        useCombined = a.useCombinedImageSamplerIndex != VK_FALSE;
        samplerHeapOffset = a.samplerHeapOffset;
        samplerPushOffset = a.samplerPushOffset;
        samplerAddressOffset = a.samplerAddressOffset;
        samplerIndexStride = a.samplerHeapIndexStride;
        samplerArrayStride = a.samplerHeapArrayStride;
      }

      if(embedded && isSampler)
      {
        ret.kind = HeapMappingKind::EmbeddedSampler;
        break;
      }

      // the array form indexes the array of indices in memory by the shader index
      const uint64_t indexOffset = isArray ? shaderIndex * sizeof(uint32_t) : 0;

      const bool packed = isCombined && useCombined;

      uint32_t indirectIndex = 0;
      const bool ok = ReadIndirect(pushData, pushDataSize, readMemory, pushOffset,
                                   (addressOffset & ~3U) + indexOffset, indirectIndex);

      uint32_t samplerIndex = 0;
      bool samplerOk = ok;

      if(packed)
      {
        samplerIndex = (indirectIndex >> 20) & 0xFFF;
        indirectIndex &= 0xFFFFF;
      }
      else if(isCombined && !embedded)
      {
        samplerOk = ReadIndirect(pushData, pushDataSize, readMemory, samplerPushOffset,
                                 (samplerAddressOffset & ~3U) + indexOffset, samplerIndex);
      }

      if(ok)
      {
        ret.kind = mainHeap;
        ret.byteOffset = uint64_t(heapOffset) + uint64_t(indirectIndex) * indexStride +
                         (isArray ? 0 : shaderIndex * arrayStride);
      }
      else
      {
        ret.kind = HeapMappingKind::Unresolved;
      }

      if(isCombined)
      {
        if(embedded)
        {
          ret.samplerKind = HeapMappingKind::EmbeddedSampler;
        }
        else if(samplerOk)
        {
          ret.samplerKind = HeapMappingKind::SamplerHeap;
          ret.samplerByteOffset = uint64_t(samplerHeapOffset) +
                                  uint64_t(samplerIndex) * samplerIndexStride +
                                  (isArray ? 0 : shaderIndex * samplerArrayStride);
        }
        else
        {
          ret.samplerKind = HeapMappingKind::Unresolved;
        }
      }
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT:
    {
      // indirectAddress = push data[pushOffset], resourceAddress = memory[indirectAddress + addressOffset]
      uint64_t resourceAddress = 0;
      if(ReadIndirect(pushData, pushDataSize, readMemory, src.indirectAddress.pushOffset,
                      src.indirectAddress.addressOffset & ~7U, resourceAddress))
      {
        ret.kind = HeapMappingKind::IndirectAddress;
        ret.byteOffset = src.indirectAddress.pushOffset;
        ret.address = resourceAddress;
      }
      else
      {
        ret.kind = HeapMappingKind::Unresolved;
      }
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT:
    case VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_DATA_EXT:
    case VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT:
    {
      // shader records are ray tracing only, which isn't supported
      ret.kind = HeapMappingKind::Unresolved;

      if(isCombined)
      {
        // an embedded sampler is still known statically for the index sources
        ret.samplerKind = embedded ? HeapMappingKind::EmbeddedSampler : HeapMappingKind::Unresolved;
      }
      else if(embedded)
      {
        ret.kind = HeapMappingKind::EmbeddedSampler;
      }
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT:
    {
      uint32_t pushValue = 0;
      if(ReadPushU32(pushData, pushDataSize, src.heapData.pushOffset & ~3U, pushValue))
      {
        ret.kind = HeapMappingKind::HeapData;
        ret.byteOffset = uint64_t(src.heapData.heapOffset) + pushValue;
      }
      else
      {
        ret.kind = HeapMappingKind::Unresolved;
      }
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT:
    {
      ret.kind = HeapMappingKind::PushData;
      ret.byteOffset = src.pushDataOffset;
      break;
    }
    case VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT:
    {
      ret.kind = HeapMappingKind::PushAddress;
      ret.byteOffset = src.pushAddressOffset;
      // the address is optional information, the offset is always valid
      ReadPushU64(pushData, pushDataSize, src.pushAddressOffset & ~7U, ret.address);
      break;
    }
    default: break;
  }

  return ret;
}
