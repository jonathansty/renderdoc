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
      "offsets, push indices and embedded samplers, as well as direct heap indexing from "
      "Slang-compiled shaders.";

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

  // shaders that index the heaps directly with SPV_EXT_descriptor_heap, without any set/binding
  // mappings. These are compiled from Slang (glslang can't generate direct heap access) with:
  //
  // slangc -target spirv -profile spirv_1_4 -capability spvDescriptorHeapEXT -emit-spirv-directly
  //        -O2 -g0 -entry <entry> -stage <stage>
  //
  // Compute shader (entry main):
  /*
struct ComputeHandles
{
  DescriptorHandle<Texture2D> tex;
  DescriptorHandle<SamplerState> samp;
  DescriptorHandle<StructuredBuffer<float4>> inBuf;
  DescriptorHandle<RWStructuredBuffer<float4>> outBuf;
  DescriptorHandle<RWTexture2D<float4>> outTex;
};

[[vk::push_constant]] ComputeHandles handles;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  float4 col = handles.tex.SampleLevel(handles.samp, float2(0.5f, 0.5f), 0) + handles.inBuf[0];
  handles.outBuf[id.x] = col;
  handles.outTex[id.xy] = col;
}
  */
  // Graphics shaders (entries vsmain and psmain):
  /*
struct DrawHandles
{
  DescriptorHandle<StructuredBuffer<float4>> buf;
  DescriptorHandle<Texture2D> tex;
  DescriptorHandle<SamplerState> samp;
};

[[vk::push_constant]] DrawHandles handles;

struct V2F
{
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

[shader("vertex")]
V2F vsmain(uint vid : SV_VulkanVertexID)
{
  float2 positions[4] = {
    float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(-1.0f, 1.0f), float2(1.0f, 1.0f),
  };

  V2F ret;
  // buf[1] holds a scale for the quad
  ret.pos = float4(positions[vid] * handles.buf[1].xy, 0.0f, 1.0f);
  ret.uv = positions[vid] * 0.5f + 0.5f;
  return ret;
}

[shader("fragment")]
float4 psmain(V2F IN) : SV_Target0
{
  // buf[0] holds a colour multiplier
  return handles.tex.Sample(handles.samp, IN.uv) * handles.buf[0];
}
  */

  const std::vector<uint32_t> directCompSpirv = {
      0x07230203, 0x00010400, 0x00280000, 0x00000052, 0x00000000, 0x00020011, 0x00001179,
      0x00020011, 0x00001408, 0x00020011, 0x00000037, 0x00020011, 0x00000038, 0x00020011,
      0x00000001, 0x0008000a, 0x5f565053, 0x5f52484b, 0x79746e75, 0x5f646570, 0x6e696f70,
      0x73726574, 0x00000000, 0x0007000a, 0x5f565053, 0x5f545845, 0x63736564, 0x74706972,
      0x685f726f, 0x00706165, 0x000b000a, 0x5f565053, 0x5f52484b, 0x726f7473, 0x5f656761,
      0x66667562, 0x735f7265, 0x61726f74, 0x635f6567, 0x7373616c, 0x00000000, 0x0003000e,
      0x00000000, 0x00000001, 0x0009000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000,
      0x00000002, 0x00000003, 0x00000004, 0x00000005, 0x00060010, 0x00000001, 0x00000011,
      0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x0000000b, 0x00000001, 0x00080005,
      0x00000006, 0x706d6f43, 0x48657475, 0x6c646e61, 0x735f7365, 0x33346474, 0x00000030,
      0x00040006, 0x00000006, 0x00000000, 0x00786574, 0x00050006, 0x00000006, 0x00000001,
      0x706d6173, 0x00000000, 0x00050006, 0x00000006, 0x00000002, 0x75426e69, 0x00000066,
      0x00050006, 0x00000006, 0x00000003, 0x4274756f, 0x00006675, 0x00050006, 0x00000006,
      0x00000004, 0x5474756f, 0x00007865, 0x00040005, 0x00000004, 0x646e6168, 0x0073656c,
      0x00070005, 0x00000002, 0x6e616c73, 0x65725f67, 0x72756f73, 0x65486563, 0x00007061,
      0x00070005, 0x00000003, 0x6e616c73, 0x61735f67, 0x656c706d, 0x61654872, 0x00000070,
      0x00050005, 0x00000007, 0x61735f5f, 0x656c706d, 0x00000064, 0x00070005, 0x00000008,
      0x75727453, 0x72757463, 0x75426465, 0x72656666, 0x00000000, 0x00060006, 0x00000008,
      0x00000000, 0x656d5f5f, 0x7265626d, 0x00000030, 0x00030005, 0x00000009, 0x006c6f63,
      0x00070005, 0x0000000a, 0x74535752, 0x74637572, 0x64657275, 0x66667542, 0x00007265,
      0x00060006, 0x0000000a, 0x00000000, 0x656d5f5f, 0x7265626d, 0x00000030, 0x00040005,
      0x00000001, 0x6e69616d, 0x00000000, 0x00030047, 0x00000006, 0x00000002, 0x00050048,
      0x00000006, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000006, 0x00000001,
      0x00000023, 0x00000008, 0x00050048, 0x00000006, 0x00000002, 0x00000023, 0x00000010,
      0x00050048, 0x00000006, 0x00000003, 0x00000023, 0x00000018, 0x00050048, 0x00000006,
      0x00000004, 0x00000023, 0x00000020, 0x0004014c, 0x0000000b, 0x00001404, 0x0000000c,
      0x00040047, 0x00000002, 0x0000000b, 0x00001403, 0x0004014c, 0x0000000d, 0x00001404,
      0x0000000e, 0x00040047, 0x00000003, 0x0000000b, 0x00001402, 0x0004014c, 0x0000000f,
      0x00001404, 0x00000010, 0x00040047, 0x00000011, 0x00000006, 0x00000010, 0x00030047,
      0x00000008, 0x00000002, 0x00050048, 0x00000008, 0x00000000, 0x00000023, 0x00000000,
      0x00030047, 0x00000012, 0x00000018, 0x00040047, 0x00000013, 0x00000006, 0x00000010,
      0x00030047, 0x0000000a, 0x00000002, 0x00050048, 0x0000000a, 0x00000000, 0x00000023,
      0x00000000, 0x00040047, 0x00000005, 0x0000000b, 0x0000001c, 0x0004014c, 0x00000014,
      0x00001404, 0x00000015, 0x00020013, 0x00000016, 0x00030021, 0x00000017, 0x00000016,
      0x00040015, 0x00000018, 0x00000020, 0x00000000, 0x00040017, 0x00000019, 0x00000018,
      0x00000002, 0x0007001e, 0x00000006, 0x00000019, 0x00000019, 0x00000019, 0x00000019,
      0x00000019, 0x00040020, 0x0000001a, 0x00000009, 0x00000006, 0x00040015, 0x0000001b,
      0x00000020, 0x00000001, 0x0004002b, 0x0000001b, 0x0000001c, 0x00000000, 0x00040020,
      0x0000001d, 0x00000009, 0x00000019, 0x00030016, 0x0000001e, 0x00000020, 0x00090019,
      0x0000001f, 0x0000001e, 0x00000001, 0x00000002, 0x00000000, 0x00000000, 0x00000001,
      0x00000000, 0x00041409, 0x00000018, 0x0000000c, 0x0000001f, 0x0003001d, 0x0000000b,
      0x0000001f, 0x00031141, 0x00000020, 0x00000000, 0x0004002b, 0x0000001b, 0x00000021,
      0x00000001, 0x0002001a, 0x00000022, 0x00041409, 0x00000018, 0x0000000e, 0x00000022,
      0x0003001d, 0x0000000d, 0x00000022, 0x0003001b, 0x00000023, 0x0000001f, 0x00040017,
      0x00000024, 0x0000001e, 0x00000004, 0x00040017, 0x00000025, 0x0000001e, 0x00000002,
      0x0004002b, 0x0000001e, 0x00000026, 0x3f000000, 0x0005002c, 0x00000025, 0x00000027,
      0x00000026, 0x00000026, 0x0004002b, 0x0000001e, 0x00000028, 0x00000000, 0x0004002b,
      0x0000001b, 0x00000029, 0x00000002, 0x000313fb, 0x0000002a, 0x0000000c, 0x00041409,
      0x00000018, 0x00000010, 0x0000002a, 0x0003001d, 0x0000000f, 0x0000002a, 0x0003001d,
      0x00000011, 0x00000024, 0x0003001e, 0x00000008, 0x00000011, 0x00040020, 0x0000002b,
      0x0000000c, 0x00000008, 0x00040020, 0x00000013, 0x0000000c, 0x00000024, 0x0004002b,
      0x0000001b, 0x0000002c, 0x00000003, 0x0003001e, 0x0000000a, 0x00000011, 0x00040020,
      0x0000002d, 0x0000000c, 0x0000000a, 0x00040017, 0x0000002e, 0x00000018, 0x00000003,
      0x00040020, 0x0000002f, 0x00000001, 0x0000002e, 0x0004002b, 0x0000001b, 0x00000030,
      0x00000004, 0x00090019, 0x00000031, 0x0000001e, 0x00000001, 0x00000002, 0x00000000,
      0x00000000, 0x00000002, 0x00000000, 0x00041409, 0x00000018, 0x00000015, 0x00000031,
      0x0003001d, 0x00000014, 0x00000031, 0x0004003b, 0x0000001a, 0x00000004, 0x00000009,
      0x00041142, 0x00000020, 0x00000002, 0x00000000, 0x00041142, 0x00000020, 0x00000003,
      0x00000000, 0x0004003b, 0x0000002f, 0x00000005, 0x00000001, 0x00050036, 0x00000016,
      0x00000001, 0x00000000, 0x00000017, 0x000200f8, 0x00000032, 0x00050041, 0x0000001d,
      0x00000033, 0x00000004, 0x0000001c, 0x0004003d, 0x00000019, 0x00000034, 0x00000033,
      0x00050051, 0x00000018, 0x00000035, 0x00000034, 0x00000000, 0x00061143, 0x00000020,
      0x00000036, 0x0000000b, 0x00000002, 0x00000035, 0x0004003d, 0x0000001f, 0x00000037,
      0x00000036, 0x00050041, 0x0000001d, 0x00000038, 0x00000004, 0x00000021, 0x0004003d,
      0x00000019, 0x00000039, 0x00000038, 0x00050051, 0x00000018, 0x0000003a, 0x00000039,
      0x00000000, 0x00061143, 0x00000020, 0x0000003b, 0x0000000d, 0x00000003, 0x0000003a,
      0x0004003d, 0x00000022, 0x0000003c, 0x0000003b, 0x00050056, 0x00000023, 0x0000003d,
      0x00000037, 0x0000003c, 0x00070058, 0x00000024, 0x00000007, 0x0000003d, 0x00000027,
      0x00000002, 0x00000028, 0x00050041, 0x0000001d, 0x0000003e, 0x00000004, 0x00000029,
      0x0004003d, 0x00000019, 0x0000003f, 0x0000003e, 0x00050051, 0x00000018, 0x00000040,
      0x0000003f, 0x00000000, 0x00061143, 0x00000020, 0x00000041, 0x0000000f, 0x00000002,
      0x00000040, 0x000413ff, 0x0000002b, 0x00000012, 0x00000041, 0x00060041, 0x00000013,
      0x00000042, 0x00000012, 0x0000001c, 0x0000001c, 0x0004003d, 0x00000024, 0x00000043,
      0x00000042, 0x00050081, 0x00000024, 0x00000009, 0x00000007, 0x00000043, 0x00050041,
      0x0000001d, 0x00000044, 0x00000004, 0x0000002c, 0x0004003d, 0x00000019, 0x00000045,
      0x00000044, 0x00050051, 0x00000018, 0x00000046, 0x00000045, 0x00000000, 0x00061143,
      0x00000020, 0x00000047, 0x0000000f, 0x00000002, 0x00000046, 0x000413ff, 0x0000002d,
      0x00000048, 0x00000047, 0x0004003d, 0x0000002e, 0x00000049, 0x00000005, 0x00050051,
      0x00000018, 0x0000004a, 0x00000049, 0x00000000, 0x00060041, 0x00000013, 0x0000004b,
      0x00000048, 0x0000001c, 0x0000004a, 0x0003003e, 0x0000004b, 0x00000009, 0x00050041,
      0x0000001d, 0x0000004c, 0x00000004, 0x00000030, 0x0004003d, 0x00000019, 0x0000004d,
      0x0000004c, 0x00050051, 0x00000018, 0x0000004e, 0x0000004d, 0x00000000, 0x00061143,
      0x00000020, 0x0000004f, 0x00000014, 0x00000002, 0x0000004e, 0x0004003d, 0x00000031,
      0x00000050, 0x0000004f, 0x0007004f, 0x00000019, 0x00000051, 0x00000049, 0x00000049,
      0x00000000, 0x00000001, 0x00040063, 0x00000050, 0x00000051, 0x00000009, 0x000100fd,
      0x00010038,
  };

  const std::vector<uint32_t> directVertSpirv = {
      0x07230203, 0x00010400, 0x00280000, 0x00000042, 0x00000000, 0x00020011, 0x00001179,
      0x00020011, 0x00001408, 0x00020011, 0x00000001, 0x0008000a, 0x5f565053, 0x5f52484b,
      0x79746e75, 0x5f646570, 0x6e696f70, 0x73726574, 0x00000000, 0x0007000a, 0x5f565053,
      0x5f545845, 0x63736564, 0x74706972, 0x685f726f, 0x00706165, 0x000b000a, 0x5f565053,
      0x5f52484b, 0x726f7473, 0x5f656761, 0x66667562, 0x735f7265, 0x61726f74, 0x635f6567,
      0x7373616c, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x000a000f, 0x00000000,
      0x00000001, 0x6e69616d, 0x00000000, 0x00000002, 0x00000003, 0x00000004, 0x00000005,
      0x00000006, 0x00030003, 0x0000000b, 0x00000001, 0x00070005, 0x00000007, 0x77617244,
      0x646e6148, 0x5f73656c, 0x34647473, 0x00003033, 0x00040006, 0x00000007, 0x00000000,
      0x00667562, 0x00040006, 0x00000007, 0x00000001, 0x00786574, 0x00050006, 0x00000007,
      0x00000002, 0x706d6173, 0x00000000, 0x00040005, 0x00000003, 0x646e6168, 0x0073656c,
      0x00070005, 0x00000002, 0x6e616c73, 0x65725f67, 0x72756f73, 0x65486563, 0x00007061,
      0x00070005, 0x00000008, 0x75727453, 0x72757463, 0x75426465, 0x72656666, 0x00000000,
      0x00060006, 0x00000008, 0x00000000, 0x656d5f5f, 0x7265626d, 0x00000030, 0x00090005,
      0x00000005, 0x72746e65, 0x696f5079, 0x6150746e, 0x5f6d6172, 0x616d7376, 0x752e6e69,
      0x00000076, 0x00040005, 0x00000001, 0x616d7376, 0x00006e69, 0x00040047, 0x00000006,
      0x0000000b, 0x0000002a, 0x00030047, 0x00000007, 0x00000002, 0x00050048, 0x00000007,
      0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000007, 0x00000001, 0x00000023,
      0x00000008, 0x00050048, 0x00000007, 0x00000002, 0x00000023, 0x00000010, 0x0004014c,
      0x00000009, 0x00001404, 0x0000000a, 0x00040047, 0x00000002, 0x0000000b, 0x00001403,
      0x00040047, 0x0000000b, 0x00000006, 0x00000010, 0x00030047, 0x00000008, 0x00000002,
      0x00050048, 0x00000008, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000000c,
      0x00000018, 0x00040047, 0x0000000d, 0x00000006, 0x00000010, 0x00040047, 0x00000004,
      0x0000000b, 0x00000000, 0x00040047, 0x00000005, 0x0000001e, 0x00000000, 0x00020013,
      0x0000000e, 0x00030021, 0x0000000f, 0x0000000e, 0x00030016, 0x00000010, 0x00000020,
      0x00040017, 0x00000011, 0x00000010, 0x00000004, 0x00040017, 0x00000012, 0x00000010,
      0x00000002, 0x00040015, 0x00000013, 0x00000020, 0x00000001, 0x0004002b, 0x00000013,
      0x00000014, 0x00000004, 0x0004001c, 0x00000015, 0x00000012, 0x00000014, 0x00040020,
      0x00000016, 0x00000007, 0x00000015, 0x0004002b, 0x00000013, 0x00000017, 0x00000000,
      0x00040020, 0x00000018, 0x00000001, 0x00000013, 0x00040015, 0x00000019, 0x00000020,
      0x00000000, 0x0004002b, 0x00000010, 0x0000001a, 0xbf800000, 0x0005002c, 0x00000012,
      0x0000001b, 0x0000001a, 0x0000001a, 0x0004002b, 0x00000010, 0x0000001c, 0x3f800000,
      0x0005002c, 0x00000012, 0x0000001d, 0x0000001c, 0x0000001a, 0x0005002c, 0x00000012,
      0x0000001e, 0x0000001a, 0x0000001c, 0x0005002c, 0x00000012, 0x0000001f, 0x0000001c,
      0x0000001c, 0x0007002c, 0x00000015, 0x00000020, 0x0000001b, 0x0000001d, 0x0000001e,
      0x0000001f, 0x00040020, 0x00000021, 0x00000007, 0x00000012, 0x00040017, 0x00000022,
      0x00000019, 0x00000002, 0x0005001e, 0x00000007, 0x00000022, 0x00000022, 0x00000022,
      0x00040020, 0x00000023, 0x00000009, 0x00000007, 0x00040020, 0x00000024, 0x00000009,
      0x00000022, 0x000313fb, 0x00000025, 0x0000000c, 0x00041409, 0x00000019, 0x0000000a,
      0x00000025, 0x0003001d, 0x00000009, 0x00000025, 0x00031141, 0x00000026, 0x00000000,
      0x0003001d, 0x0000000b, 0x00000011, 0x0003001e, 0x00000008, 0x0000000b, 0x00040020,
      0x00000027, 0x0000000c, 0x00000008, 0x0004002b, 0x00000013, 0x00000028, 0x00000001,
      0x00040020, 0x0000000d, 0x0000000c, 0x00000011, 0x0004002b, 0x00000010, 0x00000029,
      0x00000000, 0x0004002b, 0x00000010, 0x0000002a, 0x3f000000, 0x00040020, 0x0000002b,
      0x00000003, 0x00000011, 0x00040020, 0x0000002c, 0x00000003, 0x00000012, 0x0004003b,
      0x00000018, 0x00000006, 0x00000001, 0x0004003b, 0x00000023, 0x00000003, 0x00000009,
      0x00041142, 0x00000026, 0x00000002, 0x00000000, 0x0004003b, 0x0000002b, 0x00000004,
      0x00000003, 0x0004003b, 0x0000002c, 0x00000005, 0x00000003, 0x0005002c, 0x00000012,
      0x0000002d, 0x0000002a, 0x0000002a, 0x00050036, 0x0000000e, 0x00000001, 0x00000000,
      0x0000000f, 0x000200f8, 0x0000002e, 0x0004003b, 0x00000016, 0x0000002f, 0x00000007,
      0x0004003b, 0x00000016, 0x00000030, 0x00000007, 0x0004003d, 0x00000013, 0x00000031,
      0x00000006, 0x0004007c, 0x00000019, 0x00000032, 0x00000031, 0x0003003e, 0x0000002f,
      0x00000020, 0x00050041, 0x00000021, 0x00000033, 0x0000002f, 0x00000032, 0x0004003d,
      0x00000012, 0x00000034, 0x00000033, 0x00050041, 0x00000024, 0x00000035, 0x00000003,
      0x00000017, 0x0004003d, 0x00000022, 0x00000036, 0x00000035, 0x00050051, 0x00000019,
      0x00000037, 0x00000036, 0x00000000, 0x00061143, 0x00000026, 0x00000038, 0x00000009,
      0x00000002, 0x00000037, 0x000413ff, 0x00000027, 0x0000000c, 0x00000038, 0x00060041,
      0x0000000d, 0x00000039, 0x0000000c, 0x00000017, 0x00000028, 0x0004003d, 0x00000011,
      0x0000003a, 0x00000039, 0x0007004f, 0x00000012, 0x0000003b, 0x0000003a, 0x0000003a,
      0x00000000, 0x00000001, 0x00050085, 0x00000012, 0x0000003c, 0x00000034, 0x0000003b,
      0x00060050, 0x00000011, 0x0000003d, 0x0000003c, 0x00000029, 0x0000001c, 0x0003003e,
      0x00000030, 0x00000020, 0x00050041, 0x00000021, 0x0000003e, 0x00000030, 0x00000032,
      0x0004003d, 0x00000012, 0x0000003f, 0x0000003e, 0x0005008e, 0x00000012, 0x00000040,
      0x0000003f, 0x0000002a, 0x00050081, 0x00000012, 0x00000041, 0x00000040, 0x0000002d,
      0x0003003e, 0x00000004, 0x0000003d, 0x0003003e, 0x00000005, 0x00000041, 0x000100fd,
      0x00010038,
  };

  const std::vector<uint32_t> directFragSpirv = {
      0x07230203, 0x00010400, 0x00280000, 0x0000003c, 0x00000000, 0x00020011, 0x00001179,
      0x00020011, 0x00001408, 0x00020011, 0x00000001, 0x0008000a, 0x5f565053, 0x5f52484b,
      0x79746e75, 0x5f646570, 0x6e696f70, 0x73726574, 0x00000000, 0x0007000a, 0x5f565053,
      0x5f545845, 0x63736564, 0x74706972, 0x685f726f, 0x00706165, 0x000b000a, 0x5f565053,
      0x5f52484b, 0x726f7473, 0x5f656761, 0x66667562, 0x735f7265, 0x61726f74, 0x635f6567,
      0x7373616c, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x000a000f, 0x00000004,
      0x00000001, 0x6e69616d, 0x00000000, 0x00000002, 0x00000003, 0x00000004, 0x00000005,
      0x00000006, 0x00030010, 0x00000001, 0x00000007, 0x00030003, 0x0000000b, 0x00000001,
      0x00070005, 0x00000007, 0x77617244, 0x646e6148, 0x5f73656c, 0x34647473, 0x00003033,
      0x00040006, 0x00000007, 0x00000000, 0x00667562, 0x00040006, 0x00000007, 0x00000001,
      0x00786574, 0x00050006, 0x00000007, 0x00000002, 0x706d6173, 0x00000000, 0x00040005,
      0x00000004, 0x646e6168, 0x0073656c, 0x00070005, 0x00000002, 0x6e616c73, 0x65725f67,
      0x72756f73, 0x65486563, 0x00007061, 0x00070005, 0x00000003, 0x6e616c73, 0x61735f67,
      0x656c706d, 0x61654872, 0x00000070, 0x00040005, 0x00000006, 0x752e4e49, 0x00000076,
      0x00050005, 0x00000008, 0x61735f5f, 0x656c706d, 0x00000064, 0x00070005, 0x00000009,
      0x75727453, 0x72757463, 0x75426465, 0x72656666, 0x00000000, 0x00060006, 0x00000009,
      0x00000000, 0x656d5f5f, 0x7265626d, 0x00000030, 0x00080005, 0x00000005, 0x72746e65,
      0x696f5079, 0x6150746e, 0x5f6d6172, 0x616d7370, 0x00006e69, 0x00040005, 0x00000001,
      0x616d7370, 0x00006e69, 0x00030047, 0x00000007, 0x00000002, 0x00050048, 0x00000007,
      0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000007, 0x00000001, 0x00000023,
      0x00000008, 0x00050048, 0x00000007, 0x00000002, 0x00000023, 0x00000010, 0x0004014c,
      0x0000000a, 0x00001404, 0x0000000b, 0x00040047, 0x00000002, 0x0000000b, 0x00001403,
      0x0004014c, 0x0000000c, 0x00001404, 0x0000000d, 0x00040047, 0x00000003, 0x0000000b,
      0x00001402, 0x00040047, 0x00000006, 0x0000001e, 0x00000000, 0x0004014c, 0x0000000e,
      0x00001404, 0x0000000f, 0x00040047, 0x00000010, 0x00000006, 0x00000010, 0x00030047,
      0x00000009, 0x00000002, 0x00050048, 0x00000009, 0x00000000, 0x00000023, 0x00000000,
      0x00030047, 0x00000011, 0x00000018, 0x00040047, 0x00000012, 0x00000006, 0x00000010,
      0x00040047, 0x00000005, 0x0000001e, 0x00000000, 0x00020013, 0x00000013, 0x00030021,
      0x00000014, 0x00000013, 0x00040015, 0x00000015, 0x00000020, 0x00000000, 0x00040017,
      0x00000016, 0x00000015, 0x00000002, 0x0005001e, 0x00000007, 0x00000016, 0x00000016,
      0x00000016, 0x00040020, 0x00000017, 0x00000009, 0x00000007, 0x00040015, 0x00000018,
      0x00000020, 0x00000001, 0x0004002b, 0x00000018, 0x00000019, 0x00000001, 0x00040020,
      0x0000001a, 0x00000009, 0x00000016, 0x00030016, 0x0000001b, 0x00000020, 0x00090019,
      0x0000001c, 0x0000001b, 0x00000001, 0x00000002, 0x00000000, 0x00000000, 0x00000001,
      0x00000000, 0x00041409, 0x00000015, 0x0000000b, 0x0000001c, 0x0003001d, 0x0000000a,
      0x0000001c, 0x00031141, 0x0000001d, 0x00000000, 0x0004002b, 0x00000018, 0x0000001e,
      0x00000002, 0x0002001a, 0x0000001f, 0x00041409, 0x00000015, 0x0000000d, 0x0000001f,
      0x0003001d, 0x0000000c, 0x0000001f, 0x00040017, 0x00000020, 0x0000001b, 0x00000002,
      0x00040020, 0x00000021, 0x00000001, 0x00000020, 0x0003001b, 0x00000022, 0x0000001c,
      0x00040017, 0x00000023, 0x0000001b, 0x00000004, 0x0004002b, 0x00000018, 0x00000024,
      0x00000000, 0x000313fb, 0x00000025, 0x0000000c, 0x00041409, 0x00000015, 0x0000000f,
      0x00000025, 0x0003001d, 0x0000000e, 0x00000025, 0x0003001d, 0x00000010, 0x00000023,
      0x0003001e, 0x00000009, 0x00000010, 0x00040020, 0x00000026, 0x0000000c, 0x00000009,
      0x00040020, 0x00000012, 0x0000000c, 0x00000023, 0x00040020, 0x00000027, 0x00000003,
      0x00000023, 0x0004003b, 0x00000017, 0x00000004, 0x00000009, 0x00041142, 0x0000001d,
      0x00000002, 0x00000000, 0x00041142, 0x0000001d, 0x00000003, 0x00000000, 0x0004003b,
      0x00000021, 0x00000006, 0x00000001, 0x0004003b, 0x00000027, 0x00000005, 0x00000003,
      0x00050036, 0x00000013, 0x00000001, 0x00000000, 0x00000014, 0x000200f8, 0x00000028,
      0x00050041, 0x0000001a, 0x00000029, 0x00000004, 0x00000019, 0x0004003d, 0x00000016,
      0x0000002a, 0x00000029, 0x00050051, 0x00000015, 0x0000002b, 0x0000002a, 0x00000000,
      0x00061143, 0x0000001d, 0x0000002c, 0x0000000a, 0x00000002, 0x0000002b, 0x0004003d,
      0x0000001c, 0x0000002d, 0x0000002c, 0x00050041, 0x0000001a, 0x0000002e, 0x00000004,
      0x0000001e, 0x0004003d, 0x00000016, 0x0000002f, 0x0000002e, 0x00050051, 0x00000015,
      0x00000030, 0x0000002f, 0x00000000, 0x00061143, 0x0000001d, 0x00000031, 0x0000000c,
      0x00000003, 0x00000030, 0x0004003d, 0x0000001f, 0x00000032, 0x00000031, 0x0004003d,
      0x00000020, 0x00000033, 0x00000006, 0x00050056, 0x00000022, 0x00000034, 0x0000002d,
      0x00000032, 0x00060057, 0x00000023, 0x00000008, 0x00000034, 0x00000033, 0x00000000,
      0x00050041, 0x0000001a, 0x00000035, 0x00000004, 0x00000024, 0x0004003d, 0x00000016,
      0x00000036, 0x00000035, 0x00050051, 0x00000015, 0x00000037, 0x00000036, 0x00000000,
      0x00061143, 0x0000001d, 0x00000038, 0x0000000e, 0x00000002, 0x00000037, 0x000413ff,
      0x00000026, 0x00000011, 0x00000038, 0x00060041, 0x00000012, 0x00000039, 0x00000011,
      0x00000024, 0x00000024, 0x0004003d, 0x00000023, 0x0000003a, 0x00000039, 0x00050085,
      0x00000023, 0x0000003b, 0x00000008, 0x0000003a, 0x0003003e, 0x00000005, 0x0000003b,
      0x000100fd, 0x00010038,
  };

  // a DescriptorHandle<T> in Slang is a 64-bit value, with the heap index in the lower 32 bits. The
  // heap byte offset is the index multiplied by the size of the descriptor type
  struct DescriptorHandle
  {
    uint32_t index;
    uint32_t pad;
  };

  struct DirectComputeHandles
  {
    DescriptorHandle tex;
    DescriptorHandle samp;
    DescriptorHandle inBuf;
    DescriptorHandle outBuf;
    DescriptorHandle outTex;
  };

  struct DirectDrawHandles
  {
    DescriptorHandle buf;
    DescriptorHandle tex;
    DescriptorHandle samp;
  };

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
    devExts.push_back(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);

    // RenderDoc's shader patching for feedback needs to be able to write from graphics stages
    optFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
    optFeatures.fragmentStoresAndAtomics = VK_TRUE;

    // the Slang compute shader declares these capabilities for its RWTexture2D
    features.shaderStorageImageReadWithoutFormat = VK_TRUE;
    features.shaderStorageImageWriteWithoutFormat = VK_TRUE;

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

    static VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedFeats = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
    };

    getPhysFeatures2(&untypedFeats);

    if(!untypedFeats.shaderUntypedPointers)
      Avail = "feature 'shaderUntypedPointers' not available";

    untypedFeats.pNext = (void *)devInfoNext;
    devInfoNext = &untypedFeats;
  }

  static uint64_t Align(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

  void WriteImageDescriptor(void *dst, VkImage image, VkFormat format,
                            VkDescriptorType type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
  {
    VkImageViewCreateInfo view = vkh::ImageViewCreateInfo(image, VK_IMAGE_VIEW_TYPE_2D, format);

    VkImageDescriptorInfoEXT imgInfo = {VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT};
    imgInfo.pView = &view;
    imgInfo.layout = VK_IMAGE_LAYOUT_GENERAL;

    VkResourceDescriptorInfoEXT info = {VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    info.type = type;
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

  VkImage MakeTestImage(const char *name, const Vec4f &col, VkImageUsageFlags extraUsage = 0)
  {
    static AllocatedBuffer uploadBuf(
        this, vkh::BufferCreateInfo(texSize * texSize * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        VmaAllocationCreateInfo({0, VMA_MEMORY_USAGE_CPU_TO_GPU}));

    AllocatedImage tex(
        this,
        vkh::ImageCreateInfo(texSize, texSize, 0, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 extraUsage),
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

  VkDeviceAddress MakeTestBuffer(const char *name, const Vec4f &data,
                                 const Vec4f &data2 = Vec4f())
  {
    AllocatedBuffer buf(this,
                        vkh::BufferCreateInfo(0x100, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR |
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
                        VmaAllocationCreateInfo({0, VMA_MEMORY_USAGE_CPU_TO_GPU}));
    setName(buf.buffer, name);
    Vec4f contents[2] = {data, data2};
    buf.upload(contents, sizeof(contents));

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
    // use embedded samplers). The first 0x1000 bytes are used by the set/binding mapping tests, the
    // rest by direct heap indexing
    const VkDeviceSize userHeapSize = 0x2000;

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

    // direct heap indexing: descriptors are at index * descriptor size. Images from 0x1000 and
    // buffers from 0x1800 in the resource heap, samplers from 0x1000 in the sampler heap. Each has
    // an unused descriptor between the accessed ones, to check that it's not reported as accessed
    const uint32_t imgSize = (uint32_t)heapProps.imageDescriptorSize;
    const uint32_t bufSize = (uint32_t)heapProps.bufferDescriptorSize;
    const uint32_t sampSize = (uint32_t)heapProps.samplerDescriptorSize;

    if((imgSize % heapProps.imageDescriptorAlignment) != 0 ||
       (bufSize % heapProps.bufferDescriptorAlignment) != 0 ||
       (sampSize % heapProps.samplerDescriptorAlignment) != 0)
    {
      TEST_ERROR("Descriptor sizes are not aligned, can't index heap by descriptor size");
      return 4;
    }

    const uint32_t directTexIdx = (0x1000 + imgSize - 1) / imgSize;
    const uint32_t directUnusedTexIdx = directTexIdx + 1;
    const uint32_t directOutTexIdx = directTexIdx + 2;
    const uint32_t directBufIdx = (0x1800 + bufSize - 1) / bufSize;
    const uint32_t directUnusedBufIdx = directBufIdx + 1;
    const uint32_t directOutBufIdx = directBufIdx + 2;
    const uint32_t directSampIdx = (0x1000 + sampSize - 1) / sampSize;
    const uint32_t directUnusedSampIdx = directSampIdx + 1;

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

    // direct heap indexing resources
    {
      VkImage directTex = MakeTestImage("directTex", Vec4f(1.0f, 1.0f, 0.0f, 1.0f));
      VkImage directUnusedTex = MakeTestImage("directUnusedTex", Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
      VkImage directOutTex = MakeTestImage("directOutTex", Vec4f(0.0f, 0.0f, 0.0f, 0.0f),
                                           VK_IMAGE_USAGE_STORAGE_BIT);

      // buf[0] is a colour, buf[1] a scale for the quad in the direct heap draw
      VkDeviceAddress directBuf = MakeTestBuffer("directBuf", Vec4f(1.0f, 0.5f, 1.0f, 1.0f),
                                                 Vec4f(0.5f, 0.5f, 0.0f, 0.0f));
      VkDeviceAddress directUnusedBuf = MakeTestBuffer("directUnusedBuf", Vec4f());
      VkDeviceAddress directOutBuf = MakeTestBuffer("directOutBuf", Vec4f());

      WriteImageDescriptor(resPtr + directTexIdx * imgSize, directTex, VK_FORMAT_R8G8B8A8_UNORM);
      WriteImageDescriptor(resPtr + directUnusedTexIdx * imgSize, directUnusedTex,
                           VK_FORMAT_R8G8B8A8_UNORM);
      WriteImageDescriptor(resPtr + directOutTexIdx * imgSize, directOutTex,
                           VK_FORMAT_R8G8B8A8_UNORM, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
      WriteBufferDescriptor(resPtr + directBufIdx * bufSize, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            directBuf, 0x100);
      WriteBufferDescriptor(resPtr + directUnusedBufIdx * bufSize,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, directUnusedBuf, 0x100);
      WriteBufferDescriptor(resPtr + directOutBufIdx * bufSize, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            directOutBuf, 0x100);
      WriteSamplerDescriptor(sampPtr + directSampIdx * sampSize, sampInfo);
      WriteSamplerDescriptor(sampPtr + directUnusedSampIdx * sampSize, sampInfo);
    }

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

    // direct heap indexing pipelines, with no mappings
    VkPipeline directGraphicsPipe;
    {
      VkShaderModule vs, fs;
      CHECK_VKR(
          vkCreateShaderModule(device, vkh::ShaderModuleCreateInfo(directVertSpirv), NULL, &vs));
      CHECK_VKR(
          vkCreateShaderModule(device, vkh::ShaderModuleCreateInfo(directFragSpirv), NULL, &fs));
      shaders.push_back(vs);
      shaders.push_back(fs);

      pipeCreateInfo.stages[0] =
          vkh::PipelineShaderStageCreateInfo(vs, VK_SHADER_STAGE_VERTEX_BIT, "main");
      pipeCreateInfo.stages[1] =
          vkh::PipelineShaderStageCreateInfo(fs, VK_SHADER_STAGE_FRAGMENT_BIT, "main");

      directGraphicsPipe = createGraphicsPipeline(pipeCreateInfo);
    }

    VkPipeline directComputePipe;
    {
      VkShaderModule cs;
      CHECK_VKR(
          vkCreateShaderModule(device, vkh::ShaderModuleCreateInfo(directCompSpirv), NULL, &cs));
      shaders.push_back(cs);

      vkh::ComputePipelineCreateInfo compCreateInfo(
          VK_NULL_HANDLE,
          vkh::PipelineShaderStageCreateInfo(cs, VK_SHADER_STAGE_COMPUTE_BIT, "main"));
      compCreateInfo.pNext = &flags2;

      directComputePipe = createComputePipeline(compCreateInfo);
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

      vkCmdBindResourceHeapEXT(cmd, &resBind);
      vkCmdBindSamplerHeapEXT(cmd, &sampBind);

      setMarker(cmd, "Direct Heap Indices: imgSize=" + std::to_string(imgSize) +
                         " bufSize=" + std::to_string(bufSize) +
                         " sampSize=" + std::to_string(sampSize) +
                         " tex=" + std::to_string(directTexIdx) +
                         " unusedTex=" + std::to_string(directUnusedTexIdx) +
                         " outTex=" + std::to_string(directOutTexIdx) +
                         " buf=" + std::to_string(directBufIdx) +
                         " unusedBuf=" + std::to_string(directUnusedBufIdx) +
                         " outBuf=" + std::to_string(directOutBufIdx) +
                         " samp=" + std::to_string(directSampIdx) +
                         " unusedSamp=" + std::to_string(directUnusedSampIdx));

      // compute dispatch indexing the heaps directly
      {
        DirectComputeHandles handles = {};
        handles.tex.index = directTexIdx;
        handles.samp.index = directSampIdx;
        handles.inBuf.index = directBufIdx;
        handles.outBuf.index = directOutBufIdx;
        handles.outTex.index = directOutTexIdx;

        VkPushDataInfoEXT pushInfo = {VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT};
        pushInfo.offset = 0;
        pushInfo.data.address = &handles;
        pushInfo.data.size = sizeof(handles);
        vkCmdPushDataEXT(cmd, &pushInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, directComputePipe);
        setMarker(cmd, "Direct Heap Dispatch");
        vkCmdDispatch(cmd, 1, 1, 1);
      }

      vkCmdBeginRenderPass(
          cmd, vkh::RenderPassBeginInfo(mainWindow->rp, mainWindow->GetFB(), mainWindow->scissor),
          VK_SUBPASS_CONTENTS_INLINE);

      // report the layout of the heaps so that the test can look up the descriptors we wrote
      setMarker(cmd, "Heap Layout: imgStride=" + std::to_string(imgStride) +
                         " bufStride=" + std::to_string(bufStride) +
                         " sampStride=" + std::to_string(sampStride) +
                         " tex=" + std::to_string(texOffset) + " ubo=" + std::to_string(uboOffset) +
                         " ssbo=" + std::to_string(ssboOffset) +
                         " texArray=" + std::to_string(texArrayOffset) +
                         " samp=" + std::to_string(sampOffset) +
                         " customSamp=" + std::to_string(customSampOffset));

      vkCmdSetScissor(cmd, 0, 1, &mainWindow->scissor);

      const float cellW = float(screenWidth) / 4.0f;
      const float cellH = float(screenHeight) / 3.0f;

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

      // draw indexing the heaps directly, from both the vertex and fragment shader
      {
        VkViewport v = {cellW * float(NUM_TESTS % 4), cellH * float(NUM_TESTS / 4), cellW, cellH,
                        0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &v);

        DirectDrawHandles handles = {};
        handles.buf.index = directBufIdx;
        handles.tex.index = directTexIdx;
        handles.samp.index = directSampIdx;

        VkPushDataInfoEXT pushInfo = {VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT};
        pushInfo.offset = 0;
        pushInfo.data.address = &handles;
        pushInfo.data.size = sizeof(handles);
        vkCmdPushDataEXT(cmd, &pushInfo);

        setMarker(cmd, "Direct Heap Draw");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, directGraphicsPipe);
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
