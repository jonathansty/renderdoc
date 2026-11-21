/******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Baldur Karlsson
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

#include "vk_test.h"

RD_TEST(VK_Descriptor_Heap, VulkanGraphicsTest)
{
  static constexpr const char *Description =
      "Test of EXT_descriptor_heap based bindings, using set/binding mappings with constant "
      "offsets, push indices and embedded samplers.";

  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
  };
  VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
  };

  const std::string vertex = R"EOSHADER(
#version 460 core

layout(location = 0) out vec2 uv;

void main()
{
  vec2 positions[] = {
    vec2(-1.0f,  1.0f),
    vec2( 1.0f,  1.0f),
    vec2(-1.0f, -1.0f),
    vec2( 1.0f, -1.0f),
  };

  gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
  uv = positions[gl_VertexIndex].xy*0.5f + 0.5f;
}

)EOSHADER";

  const std::string pixel = R"EOSHADER(
layout(location = 0) in vec2 uv;

layout(location = 0, index = 0) out vec4 Color;

// resource heap, constant offset
layout(set = 0, binding = 0) uniform texture2D tex;
// sampler heap, constant offset
layout(set = 0, binding = 1) uniform sampler samp;
// resource heap, constant offset
layout(set = 0, binding = 2, std140) uniform ubo_block
{
  vec4 col;
} ubo;
// sampler heap, constant offset, custom border colour
layout(set = 0, binding = 3) uniform sampler custom_samp;

// resource heap, push index
layout(set = 1, binding = 0, std430) readonly buffer ssbo_block
{
  vec4 col;
} ssbo;
// resource heap, push index
layout(set = 1, binding = 1) uniform texture2D tex_indexed;

// embedded sampler
layout(set = 2, binding = 0) uniform sampler embedded_samp;

void main()
{
#if TEST == 0
  Color = texture(sampler2D(tex, samp), uv);
#elif TEST == 1
  Color = ubo.col;
#elif TEST == 2 || TEST == 3
  Color = ssbo.col;
#elif TEST == 4 || TEST == 5
  Color = texture(sampler2D(tex_indexed, samp), uv);
#elif TEST == 6
  Color = texture(sampler2D(tex, embedded_samp), uv*2.0f);
#elif TEST == 7
  Color = texture(sampler2D(tex, custom_samp), uv*2.0f);
#endif
}

)EOSHADER";

  static const uint32_t NUM_TESTS = 8;

  static const uint32_t texSize = 4;

  struct PushData
  {
    uint32_t ssboIndex;
    uint32_t texIndex;
  };

  std::vector<AllocatedImage> images;
  std::vector<AllocatedBuffer> buffers;

  void Prepare(int argc, char **argv)
  {
    devExts.push_back(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    devExts.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    devExts.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    devExts.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);

    VulkanGraphicsTest::Prepare(argc, argv);

    if(!Avail.empty())
      return;

    getPhysFeatures2(&heapFeatures);
    getPhysProperties2(&heapProps);

    if(!heapFeatures.descriptorHeap)
      Avail = "Feature 'descriptorHeap' not available";

    heapFeatures.pNext = (void *)devInfoNext;
    devInfoNext = &heapFeatures;

    static VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufaddrFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
    };

    getPhysFeatures2(&bufaddrFeatures);

    if(!bufaddrFeatures.bufferDeviceAddress)
      Avail = "feature 'bufferDeviceAddress' not available";

    bufaddrFeatures.pNext = (void *)devInfoNext;
    devInfoNext = &bufaddrFeatures;

    static VkPhysicalDeviceMaintenance5FeaturesKHR maint5Feats = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
    };

    getPhysFeatures2(&maint5Feats);

    if(!maint5Feats.maintenance5)
      Avail = "feature 'maintenance5' not available";

    maint5Feats.pNext = (void *)devInfoNext;
    devInfoNext = &maint5Feats;

    static VkPhysicalDeviceCustomBorderColorFeaturesEXT customBorderFeats = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
    };

    getPhysFeatures2(&customBorderFeats);

    if(!customBorderFeats.customBorderColors)
      Avail = "feature 'customBorderColors' not available";

    customBorderFeats.pNext = (void *)devInfoNext;
    devInfoNext = &customBorderFeats;
  }

  static uint64_t Align(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

  void WriteImageDescriptor(void *dst, VkImage image, VkFormat format)
  {
    VkImageViewCreateInfo view = vkh::ImageViewCreateInfo(image, VK_IMAGE_VIEW_TYPE_2D, format);

    VkImageDescriptorInfoEXT imgInfo = {VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT};
    imgInfo.pView = &view;
    imgInfo.layout = VK_IMAGE_LAYOUT_GENERAL;

    VkResourceDescriptorInfoEXT info = {VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    info.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    info.data.pImage = &imgInfo;

    VkHostAddressRangeEXT host = {dst, (size_t)heapProps.imageDescriptorSize};

    CHECK_VKR(vkWriteResourceDescriptorsEXT(device, 1, &info, &host));
  }

  void WriteBufferDescriptor(void *dst, VkDescriptorType type, VkDeviceAddress address,
                             VkDeviceSize size)
  {
    VkDeviceAddressRangeEXT range = {address, size};

    VkResourceDescriptorInfoEXT info = {VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    info.type = type;
    info.data.pAddressRange = &range;

    VkHostAddressRangeEXT host = {dst, (size_t)heapProps.bufferDescriptorSize};

    CHECK_VKR(vkWriteResourceDescriptorsEXT(device, 1, &info, &host));
  }

  void WriteSamplerDescriptor(void *dst, const VkSamplerCreateInfo *info)
  {
    VkHostAddressRangeEXT host = {dst, (size_t)heapProps.samplerDescriptorSize};

    CHECK_VKR(vkWriteSamplerDescriptorsEXT(device, 1, info, &host));
  }

  VkImage MakeTestImage(const char *name, const Vec4f &col)
  {
    static AllocatedBuffer uploadBuf(
        this, vkh::BufferCreateInfo(texSize * texSize * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        VmaAllocationCreateInfo({0, VMA_MEMORY_USAGE_CPU_TO_GPU}));

    AllocatedImage tex(
        this,
        vkh::ImageCreateInfo(texSize, texSize, 0, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
        VmaAllocationCreateInfo(
            {VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, VMA_MEMORY_USAGE_GPU_ONLY}));
    setName(tex.image, name);

    uint8_t pixels[texSize * texSize * 4];
    for(uint32_t i = 0; i < texSize * texSize; i++)
    {
      pixels[i * 4 + 0] = uint8_t(col.x * 255.0f);
      pixels[i * 4 + 1] = uint8_t(col.y * 255.0f);
      pixels[i * 4 + 2] = uint8_t(col.z * 255.0f);
      pixels[i * 4 + 3] = uint8_t(col.w * 255.0f);
    }
    uploadBuf.upload(pixels);
    uploadBufferToImage(tex.image, {texSize, texSize, 1}, uploadBuf.buffer, VK_IMAGE_LAYOUT_GENERAL);

    images.push_back(tex);

    return tex.image;
  }

  VkDeviceAddress MakeTestBuffer(const char *name, const Vec4f &data)
  {
    AllocatedBuffer buf(this,
                        vkh::BufferCreateInfo(0x100, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR |
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
                        VmaAllocationCreateInfo({0, VMA_MEMORY_USAGE_CPU_TO_GPU}));
    setName(buf.buffer, name);
    buf.upload(&data, sizeof(data));

    buffers.push_back(buf);

    return buf.address;
  }

  int main()
  {
    vmaBDA = true;

    // initialise, create window, create context, etc
    if(!Init())
      return 3;

    // heap layout. The user area is at the start of the heap, followed by the reserved range which
    // must be at least the minimum size (the sampler heap needs a bigger reserved range because we
    // use embedded samplers)
    const VkDeviceSize userHeapSize = 0x1000;

    const VkDeviceSize resReservedOffset = Align(userHeapSize, heapProps.resourceHeapAlignment);
    const VkDeviceSize resReservedSize = heapProps.minResourceHeapReservedRange;
    const VkDeviceSize resHeapSize = resReservedOffset + resReservedSize;

    const VkDeviceSize sampReservedOffset = Align(userHeapSize, heapProps.samplerHeapAlignment);
    const VkDeviceSize sampReservedSize = std::max(heapProps.minSamplerHeapReservedRange,
                                                   heapProps.minSamplerHeapReservedRangeWithEmbedded);
    const VkDeviceSize sampHeapSize = sampReservedOffset + sampReservedSize;

    const VkDeviceSize imgStride =
        Align(heapProps.imageDescriptorSize, heapProps.imageDescriptorAlignment);
    const VkDeviceSize bufStride =
        Align(heapProps.bufferDescriptorSize, heapProps.bufferDescriptorAlignment);
    const VkDeviceSize sampStride =
        Align(heapProps.samplerDescriptorSize, heapProps.samplerDescriptorAlignment);

    // resource heap offsets
    const uint32_t texOffset = 0;
    const uint32_t uboOffset = (uint32_t)Align(texOffset + imgStride, 0x100);
    const uint32_t ssboOffset = (uint32_t)Align(uboOffset + bufStride, 0x100);
    const uint32_t texArrayOffset = (uint32_t)Align(ssboOffset + bufStride * 2, 0x100);

    // sampler heap offsets
    const uint32_t sampOffset = 0;
    const uint32_t customSampOffset = (uint32_t)Align(sampOffset + sampStride, 0x100);

    AllocatedBuffer resHeap(
        this,
        vkh::BufferCreateInfo(resHeapSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR |
                                               VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT),
        VmaAllocationCreateInfo({0, VMA_MEMORY_USAGE_CPU_TO_GPU}),
        (uint32_t)heapProps.resourceHeapAlignment);
    setName(resHeap.buffer, "resHeap");

    AllocatedBuffer sampHeap(
        this,
        vkh::BufferCreateInfo(sampHeapSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR |
                                                VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT),
        VmaAllocationCreateInfo({0, VMA_MEMORY_USAGE_CPU_TO_GPU}),
        (uint32_t)heapProps.samplerHeapAlignment);
    setName(sampHeap.buffer, "sampHeap");

    byte *resPtr = resHeap.map();
    byte *sampPtr = sampHeap.map();
    memset(resPtr, 0xcc, (size_t)resHeapSize);
    memset(sampPtr, 0xcc, (size_t)sampHeapSize);

    // resources
    VkImage tex0 = MakeTestImage("tex0", Vec4f(1.0f, 0.0f, 0.0f, 1.0f));
    VkImage texArray[2] = {
        MakeTestImage("texArray0", Vec4f(0.0f, 1.0f, 1.0f, 1.0f)),
        MakeTestImage("texArray1", Vec4f(1.0f, 0.0f, 1.0f, 1.0f)),
    };

    VkDeviceAddress uboAddr = MakeTestBuffer("ubo", Vec4f(0.0f, 1.0f, 0.0f, 1.0f));
    VkDeviceAddress ssboAddr[2] = {
        MakeTestBuffer("ssbo0", Vec4f(0.0f, 0.0f, 1.0f, 1.0f)),
        MakeTestBuffer("ssbo1", Vec4f(1.0f, 1.0f, 0.0f, 1.0f)),
    };

    // write the descriptors directly into the heaps
    WriteImageDescriptor(resPtr + texOffset, tex0, VK_FORMAT_R8G8B8A8_UNORM);
    WriteBufferDescriptor(resPtr + uboOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uboAddr, 0x100);
    for(uint32_t i = 0; i < 2; i++)
    {
      WriteBufferDescriptor(resPtr + ssboOffset + bufStride * i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            ssboAddr[i], 0x100);
      WriteImageDescriptor(resPtr + texArrayOffset + imgStride * i, texArray[i],
                           VK_FORMAT_R8G8B8A8_UNORM);
    }

    vkh::SamplerCreateInfo sampInfo(VK_FILTER_NEAREST);
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    WriteSamplerDescriptor(sampPtr + sampOffset, sampInfo);

    // embedded sampler, clamps to a white border so we can see it's used
    vkh::SamplerCreateInfo embeddedInfo(VK_FILTER_NEAREST);
    embeddedInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    embeddedInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    embeddedInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    embeddedInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    VkSamplerCustomBorderColorIndexCreateInfoEXT borderIndexInfo = {
        VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_INDEX_CREATE_INFO_EXT,
    };

    VkSamplerCustomBorderColorCreateInfoEXT border = {
        VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
    };

    vkh::SamplerCreateInfo customInfo = embeddedInfo;

    border.customBorderColor.float32[0] = 0.5f;
    border.customBorderColor.float32[1] = 0.25f;
    border.customBorderColor.float32[2] = 0.75f;
    border.customBorderColor.float32[3] = 1.0f;
    border.format = VK_FORMAT_R8G8B8A8_UNORM;

    CHECK_VKR(vkRegisterCustomBorderColorEXT(device, &border, VK_FALSE, &borderIndexInfo.index));

    customInfo.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
    // the border colour create info is chained as well as the index, as required by validation
    borderIndexInfo.pNext = &border;
    customInfo.next(&borderIndexInfo);

    WriteSamplerDescriptor(sampPtr + customSampOffset, customInfo);

    // mappings from set/binding to the heaps
    VkDescriptorSetAndBindingMappingEXT mappings[7] = {};
    for(VkDescriptorSetAndBindingMappingEXT &m : mappings)
    {
      m.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
      m.bindingCount = 1;
    }

    // set 0, binding 0: sampled image at a constant offset in the resource heap
    mappings[0].descriptorSet = 0;
    mappings[0].firstBinding = 0;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[0].sourceData.constantOffset.heapOffset = texOffset;

    // set 0, binding 1: sampler at a constant offset in the sampler heap
    mappings[1].descriptorSet = 0;
    mappings[1].firstBinding = 1;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = sampOffset;

    // set 0, binding 2: UBO at a constant offset in the resource heap
    mappings[2].descriptorSet = 0;
    mappings[2].firstBinding = 2;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[2].sourceData.constantOffset.heapOffset = uboOffset;

    // set 0, binding 3: custom border colour sampler at a constant offset in the sampler heap
    mappings[3].descriptorSet = 0;
    mappings[3].firstBinding = 3;
    mappings[3].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    mappings[3].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[3].sourceData.constantOffset.heapOffset = customSampOffset;

    // set 1, binding 0: SSBO selected by push data (index at push offset 0)
    mappings[4].descriptorSet = 1;
    mappings[4].firstBinding = 0;
    mappings[4].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
                               VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[4].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mappings[4].sourceData.pushIndex.heapOffset = ssboOffset;
    mappings[4].sourceData.pushIndex.pushOffset = offsetof(PushData, ssboIndex);
    mappings[4].sourceData.pushIndex.heapIndexStride = (uint32_t)bufStride;

    // set 1, binding 1: sampled image selected by push data (index at push offset 4)
    mappings[5].descriptorSet = 1;
    mappings[5].firstBinding = 1;
    mappings[5].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    mappings[5].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mappings[5].sourceData.pushIndex.heapOffset = texArrayOffset;
    mappings[5].sourceData.pushIndex.pushOffset = offsetof(PushData, texIndex);
    mappings[5].sourceData.pushIndex.heapIndexStride = (uint32_t)imgStride;

    // set 2, binding 0: embedded sampler
    mappings[6].descriptorSet = 2;
    mappings[6].firstBinding = 0;
    mappings[6].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    mappings[6].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[6].sourceData.constantOffset.pEmbeddedSampler = &embeddedInfo;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo = {
        VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    mappingInfo.mappingCount = ARRAY_COUNT(mappings);
    mappingInfo.pMappings = mappings;

    // heap pipelines have no layout and are flagged with the create flags 2 extension
    VkPipelineCreateFlags2CreateInfo flags2 = {VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO};
    flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    vkh::GraphicsPipelineCreateInfo pipeCreateInfo;

    pipeCreateInfo.layout = VK_NULL_HANDLE;
    pipeCreateInfo.renderPass = mainWindow->rp;
    pipeCreateInfo.pNext = &flags2;

    pipeCreateInfo.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    pipeCreateInfo.stages.resize(2);
    pipeCreateInfo.stages[0] =
        CompileShaderModule(vertex, ShaderLang::glsl, ShaderStage::vert, "main");

    std::vector<VkPipeline> tests;

    for(uint32_t i = 0; i < NUM_TESTS; i++)
    {
      pipeCreateInfo.stages[1] =
          CompileShaderModule("#version 460 core\n#define TEST " + std::to_string(i) + "\n" + pixel,
                              ShaderLang::glsl, ShaderStage::frag, "main");
      pipeCreateInfo.stages[1].pNext = &mappingInfo;

      tests.push_back(createGraphicsPipeline(pipeCreateInfo));
    }

    VkBindHeapInfoEXT resBind = {VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT};
    resBind.heapRange.address = resHeap.address;
    resBind.heapRange.size = resHeapSize;
    resBind.reservedRangeOffset = resReservedOffset;
    resBind.reservedRangeSize = resReservedSize;

    VkBindHeapInfoEXT sampBind = {VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT};
    sampBind.heapRange.address = sampHeap.address;
    sampBind.heapRange.size = sampHeapSize;
    sampBind.reservedRangeOffset = sampReservedOffset;
    sampBind.reservedRangeSize = sampReservedSize;

    while(Running())
    {
      VkCommandBuffer cmd = GetCommandBuffer();

      vkBeginCommandBuffer(cmd, vkh::CommandBufferBeginInfo());

      VkImage swapimg =
          StartUsingBackbuffer(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);

      vkCmdClearColorImage(cmd, swapimg, VK_IMAGE_LAYOUT_GENERAL,
                           vkh::ClearColorValue(0.2f, 0.2f, 0.2f, 1.0f), 1,
                           vkh::ImageSubresourceRange());

      vkCmdBeginRenderPass(
          cmd, vkh::RenderPassBeginInfo(mainWindow->rp, mainWindow->GetFB(), mainWindow->scissor),
          VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindResourceHeapEXT(cmd, &resBind);
      vkCmdBindSamplerHeapEXT(cmd, &sampBind);

      vkCmdSetScissor(cmd, 0, 1, &mainWindow->scissor);

      const float cellW = float(screenWidth) / 4.0f;
      const float cellH = float(screenHeight) / 2.0f;

      for(uint32_t t = 0; t < NUM_TESTS; t++)
      {
        VkViewport v = {cellW * float(t % 4), cellH * float(t / 4), cellW, cellH, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &v);

        // tests 2 & 3 select the SSBO, tests 4 & 5 select the image
        PushData push = {t == 3 ? 1U : 0U, t == 5 ? 1U : 0U};

        VkPushDataInfoEXT pushInfo = {VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT};
        pushInfo.offset = 0;
        pushInfo.data.address = &push;
        pushInfo.data.size = sizeof(push);
        vkCmdPushDataEXT(cmd, &pushInfo);

        setMarker(cmd, "Heap Test " + std::to_string(t));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tests[t]);
        vkCmdDraw(cmd, 4, 1, 0, 0);
      }

      vkCmdEndRenderPass(cmd);

      FinishUsingBackbuffer(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);

      vkEndCommandBuffer(cmd);

      Submit(0, 1, {cmd});

      Present();
    }

    resHeap.unmap();
    sampHeap.unmap();

    return 0;
  }
};

REGISTER_TEST();
