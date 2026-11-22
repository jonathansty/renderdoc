import struct
import renderdoc as rd
import rdtest


class VK_Descriptor_Heap(rdtest.TestCase):
    demos_test_name = 'VK_Descriptor_Heap'

    def get_replay_options(self):
        opts = rdtest.TestCase.get_replay_options(self)
        # patched pipelines used by feedback must not generate validation errors
        opts.apiValidation = True
        return opts

    def check_capture(self):
        # each test draws into its own viewport. Each entry is a list of (x fraction, y fraction, colour) to check
        # within that viewport. The tests that scale the UVs by 2 hit the sampler's border colour in the upper right
        # of the viewport and the texture in the lower left
        red = (1.0, 0.0, 0.0, 1.0)
        green = (0.0, 1.0, 0.0, 1.0)
        blue = (0.0, 0.0, 1.0, 1.0)
        yellow = (1.0, 1.0, 0.0, 1.0)
        cyan = (0.0, 1.0, 1.0, 1.0)
        magenta = (1.0, 0.0, 1.0, 1.0)
        white = (1.0, 1.0, 1.0, 1.0)
        custom = (0.5, 0.25, 0.75, 1.0)

        # image + sampler via constant offsets
        # UBO via constant offset
        # SSBO via push index (two different indices)
        # image via push index (two different indices)
        # embedded sampler
        # custom border colour sampler
        expected = [
            [(0.5, 0.5, red)],
            [(0.5, 0.5, green)],
            [(0.5, 0.5, blue)],
            [(0.5, 0.5, yellow)],
            [(0.5, 0.5, cyan)],
            [(0.5, 0.5, magenta)],
            [(0.25, 0.25, red), (0.75, 0.75, white)],
            [(0.25, 0.25, red), (0.75, 0.75, custom)],
        ]

        self.check_heap_descriptors()

        self.check_direct_heap_access()

        eid = 0

        for test, checks in enumerate(expected):
            draw = self.find_action(f"Heap Test {test}", eid)

            if draw is None:
                raise rdtest.TestFailureException(f"Couldn't find marker for Heap Test {test}")

            rdtest.log.begin_section(f"Heap Test {test}")

            eid = draw.eventId

            # drain messages from the replay so far, so we only see ones from this event's replay + feedback
            self.controller.GetDebugMessages()

            # selecting the event runs bindless feedback on the draw
            self.set_event(draw.nextAction.eventId, False)

            pipe = self.controller.GetVulkanPipelineState()

            for name, heap in (("resource", pipe.resourceHeap), ("sampler", pipe.samplerHeap)):
                if heap.buffer == rd.ResourceId():
                    raise rdtest.TestFailureException(f"Expected a {name} heap to be bound")

                if heap.size == 0 or heap.reservedSize == 0:
                    raise rdtest.TestFailureException(
                        f"Expected {name} heap to have a size and reserved range, got {heap.size} / {heap.reservedSize}")

                heap_name = self.get_resource(heap.buffer).name
                if heap_name != f"{'res' if name == 'resource' else 'samp'}Heap":
                    raise rdtest.TestFailureException(f"Unexpected {name} heap buffer name '{heap_name}'")

            # feedback ran with a patched copy of the heap pipeline. There are no direct heap
            # accesses reported yet, and the demo's set/binding mappings are not arrays, so no dynamic accesses
            # are expected - but it must complete and not generate any validation errors
            access = self.controller.GetDescriptorAccess()

            for a in access:
                if a.index != rd.DescriptorAccess.NoShaderBinding and a.arrayElement != 0:
                    raise rdtest.TestFailureException(
                        f"Unexpected dynamic descriptor access {a.stage} {a.type} {a.index}[{a.arrayElement}]")

            msgs = [m for m in self.controller.GetDebugMessages()
                    if m.severity in (rd.MessageSeverity.High, rd.MessageSeverity.Medium)]
            if len(msgs) != 0:
                raise rdtest.TestFailureException(
                    f"Got validation messages after feedback: {[m.description for m in msgs]}")

            rdtest.log.success(f"Heap Test {test} feedback ran with {len(access)} accesses")

            v = self.controller.GetPipelineState().GetViewport(0)

            out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

            for fx, fy, col in checks:
                x = int(v.x + v.width * fx)
                y = int(v.y + v.height * fy)

                self.check_pixel_value(out, x, y, col, eps=0.02)

            rdtest.log.success(f"Heap Test {test} output is as expected")

            rdtest.log.end_section(f"Heap Test {test}")

    def check_heap_descriptors(self):
        # the demo reports the layout of the heaps in a marker
        layout_action = self.find_action("Heap Layout:")

        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the heap layout marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        self.set_event(layout_action.eventId, False)
        # set the event to a draw so that the heaps are bound
        draw = self.find_action("Heap Test 0")
        self.set_event(draw.nextAction.eventId, False)

        pipe = self.controller.GetVulkanPipelineState()

        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        def fetch(store, offset, desc_type, size, count=1):
            r = rd.DescriptorRange()
            r.offset = offset
            r.descriptorSize = size
            r.count = count
            r.type = desc_type
            return r

        def res_name(rid):
            return self.get_resource(rid).name if rid != rd.ResourceId() else None

        img_stride = layout['imgStride']
        buf_stride = layout['bufStride']
        samp_stride = layout['sampStride']

        ranges = [
            fetch(res_heap, layout['tex'], rd.DescriptorType.Image, img_stride),
            fetch(res_heap, layout['ubo'], rd.DescriptorType.ConstantBuffer, buf_stride),
            fetch(res_heap, layout['ssbo'], rd.DescriptorType.ReadWriteBuffer, buf_stride, 2),
            fetch(res_heap, layout['texArray'], rd.DescriptorType.Image, img_stride, 2),
        ]

        descs = []
        for r in ranges:
            descs.extend(self.controller.GetDescriptors(res_heap, [r]))

        expected = [
            (rd.DescriptorType.Image, "tex0"),
            (rd.DescriptorType.ConstantBuffer, "ubo"),
            (rd.DescriptorType.ReadWriteBuffer, "ssbo0"),
            (rd.DescriptorType.ReadWriteBuffer, "ssbo1"),
            (rd.DescriptorType.Image, "texArray0"),
            (rd.DescriptorType.Image, "texArray1"),
        ]

        if len(descs) != len(expected):
            raise rdtest.TestFailureException(f"Expected {len(expected)} descriptors, got {len(descs)}")

        for d, (typ, name) in zip(descs, expected):
            got = res_name(d.resource)
            if d.type != typ or got != name:
                raise rdtest.TestFailureException(
                    f"Descriptor expected to be {typ} {name}, got {d.type} {got}")

            if typ == rd.DescriptorType.Image:
                if d.view == rd.ResourceId():
                    raise rdtest.TestFailureException(f"Image descriptor for {name} has no view")
                if d.firstMip != 0 or d.numMips != 1 or d.firstSlice != 0 or d.numSlices != 1:
                    raise rdtest.TestFailureException(
                        f"Image descriptor {name} has unexpected range {d.firstMip}/{d.numMips}/{d.firstSlice}/{d.numSlices}")
                if d.format.compType != rd.CompType.UNorm or d.format.compCount != 4:
                    raise rdtest.TestFailureException(f"Image descriptor {name} has unexpected format {d.format.Name()}")
            else:
                if d.byteSize != 0x100 or d.byteOffset != 0:
                    raise rdtest.TestFailureException(
                        f"Buffer descriptor {name} has unexpected range {d.byteOffset}/{d.byteSize}")

        # samplers
        samp_ranges = [
            fetch(samp_heap, layout['samp'], rd.DescriptorType.Sampler, samp_stride),
            fetch(samp_heap, layout['customSamp'], rd.DescriptorType.Sampler, samp_stride),
        ]

        samps = []
        for r in samp_ranges:
            samps.extend(self.controller.GetSamplerDescriptors(samp_heap, [r]))

        if samps[0].filter.minify != rd.FilterMode.Point or samps[0].filter.magnify != rd.FilterMode.Point:
            raise rdtest.TestFailureException(f"Sampler 0 has unexpected filter {samps[0].filter}")
        if samps[0].addressU != rd.AddressMode.ClampEdge:
            raise rdtest.TestFailureException(f"Sampler 0 has unexpected address mode {samps[0].addressU}")

        border = samps[1].borderColorValue.floatValue
        if samps[1].addressU != rd.AddressMode.ClampBorder or [round(x, 2) for x in border] != [0.5, 0.25, 0.75, 1.0]:
            raise rdtest.TestFailureException(
                f"Sampler 1 has unexpected border {samps[1].addressU} {list(border)}")

        # heap samplers have no application object, the replay-created one must still be named
        names = {r.resourceId: r.name for r in self.controller.GetResources()}
        for i, s in enumerate(samps):
            if not names.get(s.object, "").startswith("Heap Sampler "):
                raise rdtest.TestFailureException(f"Sampler {i} object {s.object} has no heap sampler name")

        # names of the heap locations
        locs = self.controller.GetDescriptorLocations(res_heap, [ranges[3]])
        n = layout['texArray'] // img_stride
        if [l.logicalBindName for l in locs] != [f"ResourceHeap[{n}]", f"ResourceHeap[{n + 1}]"]:
            raise rdtest.TestFailureException(f"Unexpected resource heap location names {[l.logicalBindName for l in locs]}")

        locs = self.controller.GetDescriptorLocations(samp_heap, [samp_ranges[1]])
        if locs[0].logicalBindName != f"SamplerHeap[{layout['customSamp'] // samp_stride}]":
            raise rdtest.TestFailureException(f"Unexpected sampler heap location name {locs[0].logicalBindName}")

        rdtest.log.success("Heap descriptors decoded as expected")

    def check_no_validation(self, what):
        msgs = [m for m in self.controller.GetDebugMessages()
                if m.severity in (rd.MessageSeverity.High, rd.MessageSeverity.Medium)]
        if len(msgs) != 0:
            raise rdtest.TestFailureException(
                f"Got validation messages after {what}: {[m.description for m in msgs]}")

    def check_direct_accesses(self, name, expected):
        """Check the direct heap accesses at the current event are exactly the expected set of
        (stage, type, heap, byte offset, resource name) tuples, and that their contents decode."""
        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        access = [a for a in self.controller.GetDescriptorAccess()
                  if a.index == rd.DescriptorAccess.NoShaderBinding]

        got = []
        for a in access:
            if a.descriptorStore == res_heap:
                heap = 'res'
            elif a.descriptorStore == samp_heap:
                heap = 'samp'
            else:
                raise rdtest.TestFailureException(
                    f"{name}: direct heap access {a.stage} {a.type} @ {a.byteOffset} has unexpected store "
                    f"{a.descriptorStore}")

            if a.type == rd.DescriptorType.Sampler:
                samp = self.controller.GetSamplerDescriptors(a.descriptorStore, [rd.DescriptorRange(a)])[0]
                if samp.filter.minify != rd.FilterMode.Point or samp.addressU != rd.AddressMode.ClampEdge:
                    raise rdtest.TestFailureException(f"{name}: sampler access decoded unexpectedly")
                res = None
            else:
                desc = self.controller.GetDescriptors(a.descriptorStore, [rd.DescriptorRange(a)])[0]
                if desc.type != a.type:
                    raise rdtest.TestFailureException(
                        f"{name}: descriptor at {a.byteOffset} is {desc.type}, access is {a.type}")
                res = self.get_resource(desc.resource).name if desc.resource != rd.ResourceId() else None

            got.append((a.stage, a.type, heap, a.byteOffset, res))

        if sorted(got, key=str) != sorted(expected, key=str):
            raise rdtest.TestFailureException(
                f"{name}: direct heap accesses don't match. Got: {sorted(got, key=str)} Expected: "
                f"{sorted(expected, key=str)}")

        rdtest.log.success(f"{name} reported the expected {len(got)} direct heap accesses")

        return access

    def check_direct_heap_image_debug(self, layout):
        """Debug threads of the image-only direct heap dispatch, which samples a texture with a sampler
        and writes to a storage image all accessed directly from the heaps, and check the debugger
        calculates the same values as the dispatch wrote."""
        rdtest.log.begin_section("Direct Heap Image Dispatch")

        img = layout['imgSize']
        samp = layout['sampSize']

        Image = rd.DescriptorType.Image
        RWImage = rd.DescriptorType.ReadWriteImage
        Sampler = rd.DescriptorType.Sampler

        dispatch = self.find_action("Direct Heap Image Dispatch").nextAction
        if not (dispatch.flags & rd.ActionFlags.Dispatch):
            raise rdtest.TestFailureException(f"Expected a dispatch after the marker, got {dispatch.customName}")

        self.controller.GetDebugMessages()
        self.set_event(dispatch.eventId, False)

        cs = rd.ShaderStage.Compute
        self.check_direct_accesses("Image dispatch", [
            (cs, Image, 'res', layout['tex'] * img, "directTex"),
            (cs, RWImage, 'res', layout['imgOutTex'] * img, "directImgOutTex"),
            (cs, Sampler, 'samp', layout['samp'] * samp, None),
        ])

        self.check_no_validation("direct heap image dispatch feedback")

        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if not refl.debugInfo.debuggable:
            raise rdtest.TestFailureException(
                f"Image-only direct heap dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        out_tex = [r for r in self.controller.GetResources() if r.name == "directImgOutTex"][0].resourceId

        dim = refl.dispatchThreadsDimension

        for thread in [(0, 0, 0), (1, 2, 0), (3, 3, 0), (2, 1, 0)]:
            # the texture is yellow, multiplied by the UV
            u = (thread[0] + 0.5) / dim[0]
            v = (thread[1] + 0.5) / dim[1]
            expected = [u, v, 0.0, 1.0]

            # the dispatch wrote this to an RGBA8 image
            written = list(self.controller.PickPixel(out_tex, thread[0], thread[1], rd.Subresource(),
                                                     rd.CompType.Typeless).floatValue)
            if not rdtest.value_compare(written, expected, eps=1.0 / 255.0):
                raise rdtest.TestFailureException(
                    f"Thread {thread} wrote {written} to the output, expected {expected}")

            with self.debug_thread((0, 0, 0), thread) as debug:
                cycles, variables = self.process_trace(debug.trace)

                # find the source variable 'col' that's written to the output
                debugged = None
                count = len(debug.trace.instInfo)
                for inst in range(count):
                    source_vars = [sv for sv in debug.trace.instInfo[count - 1 - inst].sourceVars
                                   if sv.name == 'col']
                    if len(source_vars) == 0:
                        continue
                    try:
                        debugged = self.evaluate_source_var(source_vars[0], variables)
                    except KeyError:
                        continue
                    except rdtest.TestFailureException:
                        continue
                    break

                if debugged is None:
                    raise rdtest.TestFailureException(f"Couldn't find source variable 'col' for thread {thread}")

                value = list(debugged.value.f32v[0:4])
                if not rdtest.value_compare(value, expected, eps=1.0E-5):
                    raise rdtest.TestFailureException(
                        f"Thread {thread} debugged 'col' is {value}, expected {expected}")

                # the loaded heap descriptors should refer directly to the descriptors in the heaps, the
                # same as the direct accesses from feedback
                direct = set()
                for var in variables.values():
                    if var.type in (rd.VarType.ReadOnlyResource, rd.VarType.ReadWriteResource,
                                    rd.VarType.Sampler) and var.IsDirectAccess():
                        acc = var.GetDirectAccess()
                        heap = 'res' if acc.descriptorStore == res_heap else (
                            'samp' if acc.descriptorStore == samp_heap else str(acc.descriptorStore))
                        direct.add((var.type, acc.type, heap, acc.byteOffset))

                expected_direct = {
                    (rd.VarType.ReadOnlyResource, Image, 'res', layout['tex'] * img),
                    (rd.VarType.ReadWriteResource, RWImage, 'res', layout['imgOutTex'] * img),
                    (rd.VarType.Sampler, Sampler, 'samp', layout['samp'] * samp),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"Thread {thread} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{sorted(expected_direct, key=str)}")

                rdtest.log.success(f"Thread {thread} debugged in {cycles} cycles, 'col' = {value} as expected")

        self.check_no_validation("direct heap image dispatch debugging")

        rdtest.log.end_section("Direct Heap Image Dispatch")

    def check_direct_heap_access(self):
        layout_action = self.find_action("Direct Heap Indices:")

        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the direct heap indices marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        img = layout['imgSize']
        buf = layout['bufSize']
        samp = layout['sampSize']

        Image = rd.DescriptorType.Image
        RWImage = rd.DescriptorType.ReadWriteImage
        RWBuffer = rd.DescriptorType.ReadWriteBuffer
        Sampler = rd.DescriptorType.Sampler

        # compute dispatch
        rdtest.log.begin_section("Direct Heap Dispatch")

        dispatch = self.find_action("Direct Heap Dispatch").nextAction
        if not (dispatch.flags & rd.ActionFlags.Dispatch):
            raise rdtest.TestFailureException(f"Expected a dispatch after the marker, got {dispatch.customName}")

        self.controller.GetDebugMessages()
        self.set_event(dispatch.eventId, False)

        cs = rd.ShaderStage.Compute
        access = self.check_direct_accesses("Dispatch", [
            (cs, Image, 'res', layout['tex'] * img, "directTex"),
            (cs, RWImage, 'res', layout['outTex'] * img, "directOutTex"),
            (cs, RWBuffer, 'res', layout['buf'] * buf, "directBuf"),
            (cs, RWBuffer, 'res', layout['outBuf'] * buf, "directOutBuf"),
            (cs, Sampler, 'samp', layout['samp'] * samp, None),
        ])

        for a in access:
            expected_size = samp if a.type == Sampler else (buf if a.type == RWBuffer else img)
            if a.byteSize != expected_size:
                raise rdtest.TestFailureException(
                    f"Direct heap access {a.type} has byteSize {a.byteSize}, expected {expected_size}")

        self.check_no_validation("direct heap dispatch feedback")

        # the dispatch output is texture + buffer colour
        out_buf = [r for r in self.controller.GetResources() if r.name == "directOutBuf"][0].resourceId
        data = struct.unpack_from("4f", self.controller.GetBufferData(out_buf, 0, 16), 0)
        if not rdtest.value_compare(list(data), [2.0, 1.5, 1.0, 2.0], eps=0.02):
            raise rdtest.TestFailureException(f"Direct heap dispatch output is {list(data)}")

        rdtest.log.success("Direct heap dispatch output is as expected")

        # this shader accesses buffers in the heap, which can't be debugged yet. It must be reported as
        # not debuggable rather than failing
        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if refl.debugInfo.debuggable or 'TypeBufferEXT' not in refl.debugInfo.debugStatus:
            raise rdtest.TestFailureException(
                f"Direct heap dispatch with buffers should not be debuggable, got "
                f"{refl.debugInfo.debuggable} '{refl.debugInfo.debugStatus}'")

        rdtest.log.success(f"Direct heap dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        rdtest.log.end_section("Direct Heap Dispatch")

        self.check_direct_heap_image_debug(layout)

        # draw, with the buffer accessed from both stages
        rdtest.log.begin_section("Direct Heap Draw")

        draw = self.find_action("Direct Heap Draw").nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        vs = rd.ShaderStage.Vertex
        ps = rd.ShaderStage.Pixel
        self.check_direct_accesses("Draw", [
            (vs, RWBuffer, 'res', layout['buf'] * buf, "directBuf"),
            (ps, RWBuffer, 'res', layout['buf'] * buf, "directBuf"),
            (ps, Image, 'res', layout['tex'] * img, "directTex"),
            (ps, Sampler, 'samp', layout['samp'] * samp, None),
        ])

        self.check_no_validation("direct heap draw feedback")

        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        # the quad is scaled to the middle half of the viewport, coloured by the texture * buffer colour
        self.check_pixel_value(out, int(v.x + v.width * 0.5), int(v.y + v.height * 0.5), [1.0, 0.5, 0.0, 1.0],
                               eps=0.02)
        self.check_pixel_value(out, int(v.x + v.width * 0.1), int(v.y + v.height * 0.1), [0.2, 0.2, 0.2, 1.0],
                               eps=0.02)

        rdtest.log.success("Direct heap draw output is as expected")

        rdtest.log.end_section("Direct Heap Draw")
