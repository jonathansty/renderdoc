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

        self.check_mapped_accesses()

        self.check_direct_heap_access()

        self.check_heap_buffers()

        self.check_heap_mesh_output()

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

            # resources mapped from set/binding into the heaps can't be resolved by the shader debugger
            # yet, so pixel debugging must be refused rather than producing wrong results
            fx, fy, _ = checks[0]
            trace = self.controller.DebugPixel(int(v.x + v.width * fx), int(v.y + v.height * fy),
                                               rd.DebugPixelInputs())
            if trace.debugger is not None:
                self.controller.FreeTrace(trace)
                raise rdtest.TestFailureException(f"Heap Test {test} pixel debugging should not be supported")
            self.controller.FreeTrace(trace)

            rdtest.log.end_section(f"Heap Test {test}")

    def check_heap_mesh_output(self):
        """The mesh viewer's VS Out data for graphics pipelines that use descriptor heaps"""
        def quad(positions, scale, posname, uvname):
            ref = {}
            for i, p in enumerate(positions):
                ref[i] = {'vtx': i, 'idx': i, posname: [p[0] * scale, p[1] * scale, 0.0, 1.0],
                          uvname: [p[0] * 0.5 + 0.5, p[1] * 0.5 + 0.5]}
            return ref

        tests = [
            # GLSL vertex shader, descriptors via set/binding mappings
            ("Heap Test 0", quad([(-1, 1), (1, 1), (-1, -1), (1, -1)], 1.0, 'gl_Position', 'uv')),
            # Slang vertex shader reading a heap buffer directly
            ("Direct Heap Draw", quad([(-1, -1), (1, -1), (-1, 1), (1, 1)], 0.5, '_Position', 'entryPointParam_vsmain.uv')),
            # vertex pulling from a structured buffer in the heap
            ("Heap Buffer Draw", {
                0: {'vtx': 0, 'idx': 0, '_Position': [-0.8, 0.8, 0.0, 1.0], 'entryPointParam_vsmain.col': [1.0, 0.0, 0.0, 1.0],
                    'entryPointParam_vsmain.uv': [0.0, 0.0]},
                1: {'vtx': 1, 'idx': 1, '_Position': [0.8, 0.8, 0.0, 1.0], 'entryPointParam_vsmain.col': [0.0, 1.0, 0.0, 1.0],
                    'entryPointParam_vsmain.uv': [1.0, 0.0]},
                2: {'vtx': 2, 'idx': 2, '_Position': [0.0, -0.8, 0.0, 1.0], 'entryPointParam_vsmain.col': [0.0, 0.0, 1.0, 1.0],
                    'entryPointParam_vsmain.uv': [0.5, 1.0]},
            }),
        ]

        for marker, ref in tests:
            rdtest.log.begin_section(f"{marker} VS Out")

            action = self.find_action(marker)
            if action is None:
                raise rdtest.TestFailureException(f"Couldn't find marker {marker}")

            self.controller.GetDebugMessages()
            self.set_event(action.nextAction.eventId, False)

            postvs = self.get_postvs(action.nextAction, rd.MeshDataStage.VSOut)
            if not postvs:
                raise rdtest.TestFailureException(f"No VS Out data for {marker}")

            self.check_mesh_data(ref, postvs)

            self.check_no_validation(f"{marker} VS Out")

            rdtest.log.end_section(f"{marker} VS Out")

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

    def check_mapped_accesses(self):
        """Static descriptor access for set/binding resources mapped to the heaps by the tests' pipelines"""
        layout_action = self.find_action("Heap Layout:")

        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the heap layout marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        img_stride = layout['imgStride']
        buf_stride = layout['bufStride']

        frag = rd.ShaderStage.Fragment
        ssbo_types = (rd.DescriptorType.Buffer, rd.DescriptorType.ReadWriteBuffer)

        # (stage, type, heap, byte offset, resource name). Samplers have no resource. The embedded sampler in test 6
        # isn't in either heap, so it is not reported yet
        expected = [
            [(frag, rd.DescriptorType.Image, 'res', layout['tex'], 'tex0'),
             (frag, rd.DescriptorType.Sampler, 'samp', layout['samp'], None)],
            [(frag, rd.DescriptorType.ConstantBuffer, 'res', layout['ubo'], 'ubo')],
            [(frag, 'ssbo', 'res', layout['ssbo'], 'ssbo0')],
            [(frag, 'ssbo', 'res', layout['ssbo'] + buf_stride, 'ssbo1')],
            [(frag, rd.DescriptorType.Image, 'res', layout['texArray'], 'texArray0'),
             (frag, rd.DescriptorType.Sampler, 'samp', layout['samp'], None)],
            [(frag, rd.DescriptorType.Image, 'res', layout['texArray'] + img_stride, 'texArray1'),
             (frag, rd.DescriptorType.Sampler, 'samp', layout['samp'], None)],
            [(frag, rd.DescriptorType.Image, 'res', layout['tex'], 'tex0')],
            [(frag, rd.DescriptorType.Image, 'res', layout['tex'], 'tex0'),
             (frag, rd.DescriptorType.Sampler, 'samp', layout['customSamp'], None)],
        ]

        eid = 0

        for test, exp in enumerate(expected):
            draw = self.find_action(f"Heap Test {test}", eid)

            if draw is None:
                raise rdtest.TestFailureException(f"Couldn't find marker for Heap Test {test}")

            eid = draw.eventId

            self.controller.GetDebugMessages()

            self.set_event(draw.nextAction.eventId, False)

            pipe = self.controller.GetVulkanPipelineState()
            heaps = {'res': pipe.resourceHeap.buffer, 'samp': pipe.samplerHeap.buffer}

            got = []
            for a in self.controller.GetDescriptorAccess():
                if a.index == rd.DescriptorAccess.NoShaderBinding:
                    continue

                if a.descriptorStore == rd.ResourceId():
                    raise rdtest.TestFailureException(f"Heap Test {test}: access {a.type} {a.index} has no store")

                # push constants and specialisation constants are not heap mapped
                if a.type == rd.DescriptorType.ConstantBuffer:
                    refl = self.controller.GetPipelineState().GetShaderReflection(a.stage)
                    if not refl.constantBlocks[a.index].bufferBacked:
                        continue

                heap = [k for k, v in heaps.items() if v == a.descriptorStore]
                if len(heap) != 1:
                    raise rdtest.TestFailureException(
                        f"Heap Test {test}: access {a.type} {a.index} has unexpected store {a.descriptorStore}")

                r = rd.DescriptorRange(a)

                if a.type == rd.DescriptorType.Sampler:
                    samp = self.controller.GetSamplerDescriptors(a.descriptorStore, [r])[0]
                    border = [round(x, 2) for x in samp.borderColorValue.floatValue]
                    if test == 7 and border != [0.5, 0.25, 0.75, 1.0]:
                        raise rdtest.TestFailureException(f"Heap Test {test}: sampler border is {border}")
                    if test != 7 and samp.addressU != rd.AddressMode.ClampEdge:
                        raise rdtest.TestFailureException(f"Heap Test {test}: sampler address is {samp.addressU}")
                    res = None
                    typ = a.type
                else:
                    desc = self.controller.GetDescriptors(a.descriptorStore, [r])[0]
                    res = self.get_resource(desc.resource).name if desc.resource != rd.ResourceId() else None
                    typ = 'ssbo' if a.type in ssbo_types else a.type

                got.append((a.stage, typ, heap[0], a.byteOffset, res))

            if sorted(got, key=str) != sorted(exp, key=str):
                raise rdtest.TestFailureException(
                    f"Heap Test {test}: mapped accesses don't match. Got: {sorted(got, key=str)} "
                    f"Expected: {sorted(exp, key=str)}")

            self.check_no_validation(f"Heap Test {test} mapped access")

            rdtest.log.success(f"Heap Test {test} reported the expected {len(got)} mapped accesses")

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

    def find_last_source_var(self, trace, variables, name):
        """Evaluate the last value of the named source variable in a processed debug trace."""
        count = len(trace.instInfo)
        for inst in range(count):
            source_vars = [sv for sv in trace.instInfo[count - 1 - inst].sourceVars if sv.name == name]
            if len(source_vars) == 0:
                continue
            try:
                return self.evaluate_source_var(source_vars[0], variables)
            except KeyError:
                continue
            except rdtest.TestFailureException:
                continue

        raise rdtest.TestFailureException(f"Couldn't find source variable '{name}'")

    def get_direct_heap_vars(self, variables, res_heap, samp_heap):
        """Get the set of (var type, descriptor type, heap, byte offset) for all debugger variables that
        refer directly to descriptors in the heaps."""
        direct = set()
        for var in variables.values():
            if var.type in (rd.VarType.ReadOnlyResource, rd.VarType.ReadWriteResource, rd.VarType.Sampler,
                            rd.VarType.ConstantBlock) and var.IsDirectAccess():
                acc = var.GetDirectAccess()
                heap = 'res' if acc.descriptorStore == res_heap else (
                    'samp' if acc.descriptorStore == samp_heap else str(acc.descriptorStore))
                direct.add((var.type, acc.type, heap, acc.byteOffset))
        return direct

    def check_direct_heap_buffer_debug(self, layout):
        """Debug threads of the direct heap dispatch, which reads a structured buffer, a constant buffer and
        a texture and writes to a structured buffer, a byte address buffer (atomically) and a storage image,
        all accessed directly from the heaps. Check the debugger calculates the same values as the dispatch
        wrote."""
        buf = layout['bufSize']

        RWBuffer = rd.DescriptorType.ReadWriteBuffer
        CBuffer = rd.DescriptorType.ConstantBuffer

        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        cs = rd.ShaderStage.Compute
        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if not refl.debugInfo.debuggable:
            raise rdtest.TestFailureException(
                f"Direct heap dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        resources = {r.name: r.resourceId for r in self.controller.GetResources()}

        # each thread writes (texture + inBuf[x % 2]) * scale + bias * x to outBuf[x]
        tex = [1.0, 1.0, 0.0, 1.0]
        in_buf = [[1.0, 0.5, 1.0, 1.0], [0.5, 0.5, 0.0, 0.0]]
        scale = [1.0, 1.0, 1.0, 0.5]
        bias = [0.25, 0.0, 0.5, 0.0]

        out_data = self.controller.GetBufferData(resources["directOutBuf"], 0, 64)

        # the counter is never reset, so it's incremented by 4 every frame. Each thread gets the value
        # from before the dispatch plus the number of threads that incremented it before
        counter = struct.unpack_from("I", self.controller.GetBufferData(resources["directCounterBuf"], 0, 4), 0)[0]
        if counter < 4 or (counter % 4) != 0:
            raise rdtest.TestFailureException(f"Direct heap dispatch counter is {counter}, expected a multiple of 4")

        dim = refl.dispatchThreadsDimension
        if list(dim) != [4, 1, 1]:
            raise rdtest.TestFailureException(f"Unexpected direct heap dispatch thread dimension {list(dim)}")

        for x in range(4):
            expected = [(tex[c] + in_buf[x % 2][c]) * scale[c] + bias[c] * x for c in range(4)]

            written = list(struct.unpack_from("4f", out_data, x * 16))
            if not rdtest.value_compare(written, expected, eps=0.01):
                raise rdtest.TestFailureException(f"Thread {x} wrote {written} to the output, expected {expected}")

            with self.debug_thread((0, 0, 0), (x, 0, 0)) as debug:
                cycles, variables = self.process_trace(debug.trace)

                col = list(self.find_last_source_var(debug.trace, variables, 'col').value.f32v[0:4])
                if not rdtest.value_compare(col, written, eps=1.0E-5):
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged 'col' is {col}, the dispatch wrote {written}")

                # the structured buffer is 0x100 bytes of float4s
                count = self.find_last_source_var(debug.trace, variables, 'count').value.u32v[0]
                if count != 0x100 // 16:
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged 'count' is {count}, expected {0x100 // 16}")

                # the counter is incremented atomically by every thread. The debugger starts from the
                # value before the dispatch, but other threads may or may not have been simulated
                prev = self.find_last_source_var(debug.trace, variables, 'prev').value.u32v[0]
                if prev < counter - 4 or prev >= counter:
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged 'prev' is {prev}, expected in [{counter - 4}, {counter})")

                # the buffers should refer directly to the descriptors in the heap, the same as the direct
                # accesses from feedback
                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = {
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['buf'] * buf),
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['outBuf'] * buf),
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['counterBuf'] * buf),
                    (rd.VarType.ConstantBlock, CBuffer, 'res', layout['constBuf'] * buf),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"Thread {x} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{sorted(expected_direct, key=str)}")

                rdtest.log.success(f"Thread {x} debugged in {cycles} cycles, 'col' = {col} as expected")

        self.check_no_validation("direct heap dispatch debugging")

    def check_direct_heap_draw_debug(self, layout):
        """Debug vertices and pixels of the direct heap draw, which read a structured buffer from the heap
        in both stages and sample a texture in the pixel shader."""
        buf = layout['bufSize']
        img = layout['imgSize']
        samp = layout['sampSize']

        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        for stage in (rd.ShaderStage.Vertex, rd.ShaderStage.Pixel):
            refl = self.controller.GetPipelineState().GetShaderReflection(stage)
            if not refl.debugInfo.debuggable:
                raise rdtest.TestFailureException(
                    f"Direct heap draw {stage} is not debuggable: {refl.debugInfo.debugStatus}")

        # the vertex shader generates a quad from the vertex index, scaled by buf[1].xy = (0.5, 0.5)
        positions = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

        for vtx in range(4):
            with self.debug_vertex(vtx, 0, vtx, 0) as debug:
                cycles, variables = self.process_trace(debug.trace)

                px, py = positions[vtx]
                expected_pos = [px * 0.5, py * 0.5, 0.0, 1.0]
                expected_uv = [px * 0.5 + 0.5, py * 0.5 + 0.5]

                # the position output has no source variable of its own, check the returned struct
                pos = list(self.find_last_source_var(debug.trace, variables, 'ret.pos').value.f32v[0:4])
                uv = list(self.find_last_source_var(debug.trace, variables, 'ret.uv').value.f32v[0:2])
                uv_var = self.find_output_source_var(debug.trace, rd.ShaderBuiltin.Undefined, 0)
                uv_out = list(self.evaluate_source_var(uv_var, variables).value.f32v[0:2])
                if uv_out != uv:
                    raise rdtest.TestFailureException(f"Vertex {vtx} uv output {uv_out} doesn't match 'ret.uv' {uv}")

                if not rdtest.value_compare(pos, expected_pos, eps=1.0E-5) or not rdtest.value_compare(
                        uv, expected_uv, eps=1.0E-5):
                    raise rdtest.TestFailureException(
                        f"Vertex {vtx} debugged position {pos} uv {uv}, expected {expected_pos} {expected_uv}")

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = (rd.VarType.ReadWriteResource, rd.DescriptorType.ReadWriteBuffer, 'res',
                                   layout['buf'] * buf)
                if expected_direct not in direct:
                    raise rdtest.TestFailureException(
                        f"Vertex {vtx} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{expected_direct}")

                rdtest.log.success(f"Vertex {vtx} debugged in {cycles} cycles, position {pos} as expected")

        v = self.controller.GetPipelineState().GetViewport(0)

        # the quad covers the middle half of the viewport
        for fx, fy in [(0.5, 0.5), (0.3, 0.6), (0.7, 0.3)]:
            x = int(v.x + v.width * fx)
            y = int(v.y + v.height * fy)

            # compares the debugged output against the real output
            self.check_debug_pixel(x, y)

            with self.debug_pixel(x, y, rd.DebugPixelInputs()) as debug:
                cycles, variables = self.process_trace(debug.trace)

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = {
                    (rd.VarType.ReadWriteResource, rd.DescriptorType.ReadWriteBuffer, 'res', layout['buf'] * buf),
                    (rd.VarType.ReadOnlyResource, rd.DescriptorType.Image, 'res', layout['tex'] * img),
                    (rd.VarType.Sampler, rd.DescriptorType.Sampler, 'samp', layout['samp'] * samp),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"Pixel {x},{y} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{sorted(expected_direct, key=str)}")

        self.check_no_validation("direct heap draw debugging")

    def check_heap_buffers(self):
        """The heap buffer demonstration: a particle simulation dispatch and a triangle drawn from a vertex
        buffer in the heap, accessing every kind of buffer through handles in push data."""
        layout_action = self.find_action("Heap Buffer Indices:")
        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the heap buffer indices marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        direct_layout = {}
        for item in self.find_action("Direct Heap Indices:").customName.split(":")[1].split():
            k, v = item.split("=")
            direct_layout[k] = int(v)

        buf = direct_layout['bufSize']
        img = direct_layout['imgSize']
        samp = direct_layout['sampSize']

        RWBuffer = rd.DescriptorType.ReadWriteBuffer
        CBuffer = rd.DescriptorType.ConstantBuffer
        Image = rd.DescriptorType.Image
        Sampler = rd.DescriptorType.Sampler

        resources = {r.name: r.resourceId for r in self.controller.GetResources()}

        # particle simulation dispatch
        rdtest.log.begin_section("Heap Buffer Dispatch")

        dispatch = self.find_action("Heap Buffer Dispatch").nextAction
        if not (dispatch.flags & rd.ActionFlags.Dispatch):
            raise rdtest.TestFailureException(f"Expected a dispatch after the marker, got {dispatch.customName}")

        self.controller.GetDebugMessages()
        self.set_event(dispatch.eventId, False)

        cs = rd.ShaderStage.Compute
        self.check_direct_accesses("Heap buffer dispatch", [
            (cs, RWBuffer, 'res', layout['particlesIn'] * buf, "particlesIn"),
            (cs, RWBuffer, 'res', layout['particlesOut'] * buf, "particlesOut"),
            (cs, CBuffer, 'res', layout['simConsts'] * buf, "simConsts"),
            (cs, RWBuffer, 'res', layout['weights'] * buf, "particleWeights"),
            (cs, RWBuffer, 'res', layout['stats'] * buf, "particleStats"),
        ])

        self.check_no_validation("heap buffer dispatch feedback")

        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if not refl.debugInfo.debuggable:
            raise rdtest.TestFailureException(f"Heap buffer dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        num = 8
        out_data = self.controller.GetBufferData(resources["particlesOut"], 0, 32 * num)

        # each particle starts at (i, 0, 0, 1) with velocity (0.5, 1, 0, 0), gravity -10 scaled by a weight of
        # 1 + 0.25 * i, dt = 0.1
        for i in range(num):
            w = 1.0 + 0.25 * i
            vel = [0.5, 1.0 - 10.0 * 0.1 * w, 0.0, 0.0]
            pos = [i + vel[0] * 0.1, vel[1] * 0.1, 0.0, 1.0]
            written = list(struct.unpack_from("8f", out_data, i * 32))
            if not rdtest.value_compare(written, pos + vel, eps=1.0E-5):
                raise rdtest.TestFailureException(f"Particle {i} is {written}, expected {pos + vel}")

        # the counter is reset before the dispatch, and incremented once per particle
        count = struct.unpack_from("I", self.controller.GetBufferData(resources["particleStats"], 0, 4), 0)[0]
        if count != num:
            raise rdtest.TestFailureException(f"Particle stats counter is {count}, expected {num}")

        rdtest.log.success("Heap buffer dispatch output is as expected")

        for x in [0, 3, 7]:
            written = list(struct.unpack_from("8f", out_data, x * 32))

            with self.debug_thread((0, 0, 0), (x, 0, 0)) as debug:
                cycles, variables = self.process_trace(debug.trace)

                pos = list(self.find_last_source_var(debug.trace, variables, 'p.pos').value.f32v[0:4])
                vel = list(self.find_last_source_var(debug.trace, variables, 'p.vel').value.f32v[0:4])
                weight = self.find_last_source_var(debug.trace, variables, 'weight').value.f32v[0]

                if not rdtest.value_compare(pos + vel, written, eps=1.0E-5):
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged particle {pos + vel}, the dispatch wrote {written}")

                if not rdtest.value_compare(weight, 1.0 + 0.25 * x, eps=1.0E-5):
                    raise rdtest.TestFailureException(f"Thread {x} debugged weight {weight}")

                # the atomic increment returns the counter before this thread's increment, which starts at
                # 0 for the dispatch
                prev = self.find_last_source_var(debug.trace, variables, 'prev').value.u32v[0]
                if prev >= num:
                    raise rdtest.TestFailureException(f"Thread {x} debugged 'prev' is {prev}, expected < {num}")

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = {
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['particlesIn'] * buf),
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['particlesOut'] * buf),
                    (rd.VarType.ConstantBlock, CBuffer, 'res', layout['simConsts'] * buf),
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['weights'] * buf),
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['stats'] * buf),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"Thread {x} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{sorted(expected_direct, key=str)}")

                rdtest.log.success(f"Thread {x} debugged in {cycles} cycles, particle {pos} {vel} as expected")

        self.check_no_validation("heap buffer dispatch debugging")

        rdtest.log.end_section("Heap Buffer Dispatch")

        # triangle drawn from a vertex buffer in the heap
        rdtest.log.begin_section("Heap Buffer Draw")

        draw = self.find_action("Heap Buffer Draw").nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        vs = rd.ShaderStage.Vertex
        ps = rd.ShaderStage.Pixel
        self.check_direct_accesses("Heap buffer draw", [
            (vs, RWBuffer, 'res', layout['verts'] * buf, "bufferDrawVerts"),
            (ps, CBuffer, 'res', layout['drawConsts'] * buf, "bufferDrawConsts"),
            (ps, Image, 'res', layout['tex'] * img, "bufferDrawTex"),
            (ps, Sampler, 'samp', layout['samp'] * samp, None),
        ])

        self.check_no_validation("heap buffer draw feedback")

        for stage in (vs, ps):
            refl = self.controller.GetPipelineState().GetShaderReflection(stage)
            if not refl.debugInfo.debuggable:
                raise rdtest.TestFailureException(
                    f"Heap buffer draw {stage} is not debuggable: {refl.debugInfo.debugStatus}")

        verts = [
            ([-0.8, 0.8, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),
            ([0.8, 0.8, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0]),
            ([0.0, -0.8, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.5, 1.0]),
        ]

        for vtx, (exp_pos, exp_col, exp_uv) in enumerate(verts):
            with self.debug_vertex(vtx, 0, vtx, 0) as debug:
                cycles, variables = self.process_trace(debug.trace)

                pos = list(self.find_last_source_var(debug.trace, variables, 'ret.pos').value.f32v[0:4])
                col = list(self.find_last_source_var(debug.trace, variables, 'ret.col').value.f32v[0:4])
                uv = list(self.find_last_source_var(debug.trace, variables, 'ret.uv').value.f32v[0:2])

                if not rdtest.value_compare(pos + col + uv, exp_pos + exp_col + exp_uv, eps=1.0E-5):
                    raise rdtest.TestFailureException(
                        f"Vertex {vtx} debugged {pos} {col} {uv}, expected {exp_pos} {exp_col} {exp_uv}")

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['verts'] * buf)
                if expected_direct not in direct:
                    raise rdtest.TestFailureException(
                        f"Vertex {vtx} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{expected_direct}")

                rdtest.log.success(f"Vertex {vtx} debugged in {cycles} cycles, position {pos} colour {col}")

        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        # the centre of the triangle is a mix of the vertex colours, tinted by (1, 0.5, 1, 1)
        for fx, fy in [(0.5, 0.633), (0.5, 0.8), (0.3, 0.85)]:
            x = int(v.x + v.width * fx)
            y = int(v.y + v.height * fy)

            picked = list(self.controller.PickPixel(out, x, y, rd.Subresource(), rd.CompType.Typeless).floatValue)
            if picked[1] > 0.51 or (picked[0] + picked[2]) < 0.1:
                raise rdtest.TestFailureException(f"Heap buffer draw pixel {x},{y} is {picked}, not a tinted triangle")

            # compares the debugged output against the real output
            self.check_debug_pixel(x, y)

            with self.debug_pixel(x, y, rd.DebugPixelInputs()) as debug:
                cycles, variables = self.process_trace(debug.trace)

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = {
                    (rd.VarType.ConstantBlock, CBuffer, 'res', layout['drawConsts'] * buf),
                    (rd.VarType.ReadOnlyResource, Image, 'res', layout['tex'] * img),
                    (rd.VarType.Sampler, Sampler, 'samp', layout['samp'] * samp),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"Pixel {x},{y} heap descriptor variables {sorted(direct, key=str)} don't include "
                        f"{sorted(expected_direct, key=str)}")

        self.check_no_validation("heap buffer draw debugging")

        rdtest.log.end_section("Heap Buffer Draw")

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
        CBuffer = rd.DescriptorType.ConstantBuffer
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
            (cs, RWBuffer, 'res', layout['counterBuf'] * buf, "directCounterBuf"),
            (cs, CBuffer, 'res', layout['constBuf'] * buf, "directConstBuf"),
            (cs, Sampler, 'samp', layout['samp'] * samp, None),
        ])

        for a in access:
            # uniform buffer descriptors can be smaller than the maximum buffer descriptor size
            if a.type == CBuffer:
                if a.byteSize == 0 or a.byteSize > buf:
                    raise rdtest.TestFailureException(
                        f"Direct heap access {a.type} has byteSize {a.byteSize}, expected at most {buf}")
                continue

            expected_size = samp if a.type == Sampler else (buf if a.type == RWBuffer else img)
            if a.byteSize != expected_size:
                raise rdtest.TestFailureException(
                    f"Direct heap access {a.type} has byteSize {a.byteSize}, expected {expected_size}")

        self.check_no_validation("direct heap dispatch feedback")

        # the first thread's output is (texture + buffer colour) * scale
        out_buf = [r for r in self.controller.GetResources() if r.name == "directOutBuf"][0].resourceId
        data = struct.unpack_from("4f", self.controller.GetBufferData(out_buf, 0, 16), 0)
        if not rdtest.value_compare(list(data), [2.0, 1.5, 1.0, 1.0], eps=0.02):
            raise rdtest.TestFailureException(f"Direct heap dispatch output is {list(data)}")

        rdtest.log.success("Direct heap dispatch output is as expected")

        self.check_direct_heap_buffer_debug(layout)

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

        self.check_direct_heap_draw_debug(layout)

        rdtest.log.end_section("Direct Heap Draw")
