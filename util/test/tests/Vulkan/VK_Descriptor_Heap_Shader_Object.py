import struct
import renderdoc as rd
import rdtest


class VK_Descriptor_Heap_Shader_Object(rdtest.TestCase):
    demos_test_name = 'VK_Descriptor_Heap_Shader_Object'

    def get_replay_options(self):
        opts = rdtest.TestCase.get_replay_options(self)
        # patched pipelines/shaders used by feedback must not generate validation errors
        opts.apiValidation = True
        return opts

    def check_capture(self):
        # messages generated while loading the capture are only returned by the first call, before any check
        # consumes them
        self.check_no_validation("capture load")

        self.check_shader_object_state()

        self.check_disassembly()

        self.check_layout()

        self.check_direct_draw("SO Direct Draw", True, [1.0, 0.5, 0.0, 1.0])
        self.check_mapped_draw()
        self.check_direct_draw("SO Then Pipeline Draw", False, [0.5, 1.0, 0.0, 1.0])
        self.check_direct_draw("Pipeline Then SO Draw", True, [0.25, 0.75, 0.0, 1.0])

        self.check_direct_dispatch()

        self.check_partial_replay()

        self.check_no_validation("end of test")

    def get_layout(self, marker):
        layout_action = self.find_action(marker)

        if layout_action is None:
            raise rdtest.TestFailureException(f"Couldn't find marker {marker}")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        return layout

    def flatten_actions(self, actions):
        ret = []
        for a in actions:
            ret.append(a)
            ret += self.flatten_actions(a.children)
        return ret

    def check_no_validation(self, what):
        msgs = [m for m in self.controller.GetDebugMessages()
                if m.severity in (rd.MessageSeverity.High, rd.MessageSeverity.Medium)]
        if len(msgs) != 0:
            raise rdtest.TestFailureException(
                f"Got validation messages after {what}: {[m.description for m in msgs]}")

    def check_shader_object_state(self):
        """The shader object draws and dispatch report heaps and shader objects in the pipeline state, the
        heap pipeline draw reports a pipeline"""
        for marker, objects in (("SO Direct Dispatch", True), ("SO Direct Draw", True), ("SO Mapped Draw", True),
                                ("SO Then Pipeline Draw", False), ("Pipeline Then SO Draw", True)):
            action = self.find_action(marker).nextAction

            self.set_event(action.eventId, False)

            pipe = self.controller.GetVulkanPipelineState()

            for name, heap in (("resource", pipe.resourceHeap), ("sampler", pipe.samplerHeap)):
                if heap.buffer == rd.ResourceId():
                    raise rdtest.TestFailureException(f"{marker}: expected a {name} heap to be bound")

                if heap.size == 0 or heap.reservedSize == 0:
                    raise rdtest.TestFailureException(
                        f"{marker}: expected {name} heap to have a size and reserved range, got "
                        f"{heap.size} / {heap.reservedSize}")

                heap_name = self.get_resource(heap.buffer).name
                if heap_name != f"{'res' if name == 'resource' else 'samp'}Heap":
                    raise rdtest.TestFailureException(f"{marker}: unexpected {name} heap buffer name '{heap_name}'")

            if marker == "SO Direct Dispatch":
                stage = pipe.computeShader
                if pipe.compute.pipelineResourceId != rd.ResourceId():
                    raise rdtest.TestFailureException(f"{marker}: unexpected compute pipeline")
                stages = [stage]
            else:
                stages = [pipe.vertexShader, pipe.fragmentShader]

            for stage in stages:
                if stage.resourceId == rd.ResourceId():
                    raise rdtest.TestFailureException(f"{marker}: missing shader")
                if stage.shaderObject != objects:
                    raise rdtest.TestFailureException(
                        f"{marker}: {stage.stage} shaderObject is {stage.shaderObject}, expected {objects}")

            rdtest.log.success(f"{marker}: heaps and shader objects as expected")

        self.check_no_validation("shader object state")

    def check_disassembly(self):
        """Every shader used by a draw or dispatch must disassemble, including shader objects"""
        count = 0
        for action in self.controller.GetRootActions():
            for a in [action] + self.flatten_actions(action.children):
                if not (a.flags & (rd.ActionFlags.Drawcall | rd.ActionFlags.Dispatch)):
                    continue

                self.controller.SetFrameEvent(a.eventId, False)
                pipe = self.controller.GetPipelineState()
                vkpipe = self.controller.GetVulkanPipelineState()

                compute = bool(a.flags & rd.ActionFlags.Dispatch)
                stages = [rd.ShaderStage.Compute] if compute else [rd.ShaderStage.Vertex, rd.ShaderStage.Pixel]
                pipe_obj = pipe.GetComputePipelineObject() if compute else pipe.GetGraphicsPipelineObject()

                for stage in stages:
                    refl = pipe.GetShaderReflection(stage)
                    if refl is None:
                        raise rdtest.TestFailureException(f"No reflection for {stage} at event {a.eventId}")

                    obj = pipe_obj
                    # shader objects don't have a pipeline, disassemble against the shader object itself
                    if obj == rd.ResourceId():
                        obj = pipe.GetShader(stage)

                    disasm = self.controller.DisassembleShader(obj, refl, "")
                    if disasm == "" or "error" in disasm.lower()[:200]:
                        raise rdtest.TestFailureException(
                            f"Disassembly of {stage} at event {a.eventId} failed: {disasm[:200]}")
                    count += 1

        if count != 9:
            raise rdtest.TestFailureException(f"Expected to disassemble 9 shaders, got {count}")

        rdtest.log.success(f"Disassembled {count} shaders")

    def check_layout(self):
        """The mapped descriptor layout and direct indices markers are present"""
        self.get_layout("SO Layout:")
        self.get_layout("SO Direct Indices:")

    def check_direct_accesses(self, name, expected, samp_expect=(rd.FilterMode.Point, rd.AddressMode.ClampEdge)):
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
                if (samp.filter.minify, samp.addressU) != samp_expect:
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

    def check_direct_draw(self, marker, shader_objects, centre):
        """A direct heap draw, either with shader objects or a heap pipeline, which reads a structured buffer
        from the heap in both stages and samples a texture in the pixel shader"""
        rdtest.log.begin_section(marker)

        layout = self.get_layout("SO Direct Indices:")
        buf = layout['bufSize']
        img = layout['imgSize']
        samp = layout['sampSize']

        # which buffer this draw reads
        buf_idx = {"SO Direct Draw": layout['buf'], "SO Then Pipeline Draw": layout['pipeBuf'],
                   "Pipeline Then SO Draw": layout['soAfterBuf']}[marker]
        buf_name = {"SO Direct Draw": "directBuf", "SO Then Pipeline Draw": "pipeBuf",
                    "Pipeline Then SO Draw": "soAfterBuf"}[marker]

        draw = self.find_action(marker).nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        vs = rd.ShaderStage.Vertex
        ps = rd.ShaderStage.Pixel
        RWBuffer = rd.DescriptorType.ReadWriteBuffer

        pipe = self.controller.GetVulkanPipelineState()
        if pipe.vertexShader.shaderObject != shader_objects or pipe.fragmentShader.shaderObject != shader_objects:
            raise rdtest.TestFailureException(f"{marker}: unexpected shader object state")

        self.check_direct_accesses(marker, [
            (vs, RWBuffer, 'res', buf_idx * buf, buf_name),
            (ps, RWBuffer, 'res', buf_idx * buf, buf_name),
            (ps, rd.DescriptorType.Image, 'res', layout['tex'] * img, "directTex"),
            (ps, rd.DescriptorType.Sampler, 'samp', layout['samp'] * samp, None),
        ])

        self.check_no_validation(f"{marker} feedback")

        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        # the quad is scaled to the middle half of the viewport, coloured by the texture * buffer colour
        self.check_pixel_value(out, int(v.x + v.width * 0.5), int(v.y + v.height * 0.5), centre, eps=0.02)
        self.check_pixel_value(out, int(v.x + v.width * 0.1), int(v.y + v.height * 0.1), [0.2, 0.2, 0.2, 1.0],
                               eps=0.02)

        rdtest.log.success(f"{marker} output is as expected")

        # vertex output
        postvs = self.get_postvs(draw, rd.MeshDataStage.VSOut)
        if not postvs:
            raise rdtest.TestFailureException(f"No VS Out data for {marker}")

        ref = {}
        for i, p in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            ref[i] = {'vtx': i, 'idx': i, '_Position': [p[0] * 0.5, p[1] * 0.5, 0.0, 1.0],
                      'entryPointParam_vsmain.uv': [p[0] * 0.25 + 0.5, p[1] * 0.25 + 0.5]}
            # the uv is computed from the unscaled position
            ref[i]['entryPointParam_vsmain.uv'] = [p[0] * 0.5 + 0.5, p[1] * 0.5 + 0.5]

        self.check_mesh_data(ref, postvs)

        self.check_no_validation(f"{marker} VS Out")

        # debug the vertices and pixels
        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        for stage in (vs, ps):
            refl = self.controller.GetPipelineState().GetShaderReflection(stage)
            if not refl.debugInfo.debuggable:
                raise rdtest.TestFailureException(
                    f"{marker} {stage} is not debuggable: {refl.debugInfo.debugStatus}")

        positions = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

        for vtx in range(4):
            with self.debug_vertex(vtx, 0, vtx, 0) as debug:
                cycles, variables = self.process_trace(debug.trace)

                px, py = positions[vtx]
                expected_pos = [px * 0.5, py * 0.5, 0.0, 1.0]
                expected_uv = [px * 0.5 + 0.5, py * 0.5 + 0.5]

                pos = list(self.find_last_source_var(debug.trace, variables, 'ret.pos').value.f32v[0:4])
                uv = list(self.find_last_source_var(debug.trace, variables, 'ret.uv').value.f32v[0:2])

                if not rdtest.value_compare(pos, expected_pos, eps=1.0E-5) or not rdtest.value_compare(
                        uv, expected_uv, eps=1.0E-5):
                    raise rdtest.TestFailureException(
                        f"{marker} vertex {vtx} debugged position {pos} uv {uv}, expected {expected_pos} "
                        f"{expected_uv}")

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = (rd.VarType.ReadWriteResource, RWBuffer, 'res', buf_idx * buf)
                if expected_direct not in direct:
                    raise rdtest.TestFailureException(
                        f"{marker} vertex {vtx} heap descriptor variables {sorted(direct, key=str)} don't "
                        f"include {expected_direct}")

        for fx, fy in [(0.5, 0.5), (0.3, 0.6), (0.7, 0.3)]:
            x = int(v.x + v.width * fx)
            y = int(v.y + v.height * fy)

            # compares the debugged output against the real output
            self.check_debug_pixel(x, y)

            with self.debug_pixel(x, y, rd.DebugPixelInputs()) as debug:
                cycles, variables = self.process_trace(debug.trace)

                direct = self.get_direct_heap_vars(variables, res_heap, samp_heap)
                expected_direct = {
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', buf_idx * buf),
                    (rd.VarType.ReadOnlyResource, rd.DescriptorType.Image, 'res', layout['tex'] * img),
                    (rd.VarType.Sampler, rd.DescriptorType.Sampler, 'samp', layout['samp'] * samp),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"{marker} pixel {x},{y} heap descriptor variables {sorted(direct, key=str)} don't "
                        f"include {sorted(expected_direct, key=str)}")

        self.check_no_validation(f"{marker} debugging")

        rdtest.log.end_section(marker)

    def check_mapped_draw(self):
        """Draw with shader objects using set/binding mappings chained on the shader create info, including a
        push index and an embedded sampler"""
        marker = "SO Mapped Draw"
        rdtest.log.begin_section(marker)

        layout = self.get_layout("SO Layout:")
        buf_stride = layout['bufStride']

        draw = self.find_action(marker).nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        pipe = self.controller.GetVulkanPipelineState()
        if not pipe.vertexShader.shaderObject or not pipe.fragmentShader.shaderObject:
            raise rdtest.TestFailureException(f"{marker}: expected shader objects")

        heaps = {'res': pipe.resourceHeap.buffer, 'samp': pipe.samplerHeap.buffer}
        ssbo_types = (rd.DescriptorType.Buffer, rd.DescriptorType.ReadWriteBuffer)

        frag = rd.ShaderStage.Fragment

        # (stage, type, heap, byte offset, resource name). The push index selects the second SSBO
        expected = [
            (frag, rd.DescriptorType.Image, 'res', layout['tex'], 'tex0'),
            (frag, rd.DescriptorType.Sampler, 'samp', layout['samp'], None),
            (frag, rd.DescriptorType.Sampler, 'embedded', 0, None),
            (frag, rd.DescriptorType.ConstantBuffer, 'res', layout['ubo'], 'ubo'),
            (frag, 'ssbo', 'res', layout['ssbo'] + buf_stride, 'ssbo1'),
        ]

        got = []
        for a in self.controller.GetDescriptorAccess():
            if a.index == rd.DescriptorAccess.NoShaderBinding:
                continue

            if a.descriptorStore == rd.ResourceId():
                raise rdtest.TestFailureException(f"{marker}: access {a.type} {a.index} has no store")

            # push constants and specialisation constants are not heap mapped
            if a.type == rd.DescriptorType.ConstantBuffer:
                refl = self.controller.GetPipelineState().GetShaderReflection(a.stage)
                if not refl.constantBlocks[a.index].bufferBacked:
                    continue

            heap = [k for k, v in heaps.items() if v == a.descriptorStore]
            if a.type == rd.DescriptorType.Sampler and not heap:
                heap = ['embedded']
            if len(heap) != 1:
                raise rdtest.TestFailureException(
                    f"{marker}: access {a.type} {a.index} has unexpected store {a.descriptorStore}")

            r = rd.DescriptorRange(a)

            if a.type == rd.DescriptorType.Sampler:
                samp = self.controller.GetSamplerDescriptors(a.descriptorStore, [r])[0]
                border = [round(x, 2) for x in samp.borderColorValue.floatValue]
                if heap[0] == 'embedded':
                    # embedded: nearest, clamp to opaque white border, no object, creation time constant
                    if (samp.filter.minify != rd.FilterMode.Point or samp.filter.magnify != rd.FilterMode.Point
                            or samp.addressU != rd.AddressMode.ClampBorder
                            or samp.addressV != rd.AddressMode.ClampBorder
                            or border != [1.0, 1.0, 1.0, 1.0] or samp.object != rd.ResourceId()
                            or not samp.creationTimeConstant):
                        raise rdtest.TestFailureException(f"{marker}: embedded sampler is wrong: {samp}")

                    loc = self.controller.GetDescriptorLocations(a.descriptorStore, [r])[0]
                    if loc.logicalBindName != "Embedded Sampler" or loc.category != rd.DescriptorCategory.Sampler:
                        raise rdtest.TestFailureException(
                            f"{marker}: embedded sampler location is {loc.logicalBindName}")
                elif samp.addressU != rd.AddressMode.ClampEdge:
                    raise rdtest.TestFailureException(f"{marker}: sampler address is {samp.addressU}")
                res = None
                typ = a.type
            else:
                desc = self.controller.GetDescriptors(a.descriptorStore, [r])[0]
                res = self.get_resource(desc.resource).name if desc.resource != rd.ResourceId() else None
                typ = 'ssbo' if a.type in ssbo_types else a.type

            got.append((a.stage, typ, heap[0], a.byteOffset, res))

        if sorted(got, key=str) != sorted(expected, key=str):
            raise rdtest.TestFailureException(
                f"{marker}: mapped accesses don't match. Got: {sorted(got, key=str)} "
                f"Expected: {sorted(expected, key=str)}")

        self.check_no_validation(f"{marker} mapped access")

        rdtest.log.success(f"{marker} reported the expected {len(got)} mapped accesses")

        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        # the red texture is sampled by the embedded sampler (inside the texture in the lower left of the
        # viewport, in the white border in the upper right), and by the plain sampler, summed with the UBO
        # and the second SSBO
        checks = [(0.25, 0.25, [0.875, 0.25, 0.0, 1.0]), (0.75, 0.75, [0.875, 0.75, 0.5, 1.0])]

        for fx, fy, col in checks:
            x = int(v.x + v.width * fx)
            y = int(v.y + v.height * fy)

            self.check_pixel_value(out, x, y, col, eps=0.02)

            self.check_debug_pixel(x, y)

        rdtest.log.success(f"{marker} output is as expected")

        # the vertex shader has no resources but is part of a shader object using descriptor heaps
        positions = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]

        for vtx in range(4):
            with self.debug_vertex(vtx, 0, vtx, 0) as debug:
                cycles, variables = self.process_trace(debug.trace)

                pos_var = self.find_output_source_var(debug.trace, rd.ShaderBuiltin.Position)
                pos = list(self.evaluate_source_var(pos_var, variables).value.f32v[0:4])

                expected_pos = [positions[vtx][0], positions[vtx][1], 0.0, 1.0]
                if not rdtest.value_compare(pos, expected_pos, eps=1.0E-5):
                    raise rdtest.TestFailureException(f"Vertex {vtx} debugged position {pos}, expected {expected_pos}")

        # VS Out
        postvs = self.get_postvs(draw, rd.MeshDataStage.VSOut)
        if not postvs:
            raise rdtest.TestFailureException(f"No VS Out data for {marker}")

        ref = {}
        for i, p in enumerate(positions):
            ref[i] = {'vtx': i, 'idx': i, 'gl_Position': [p[0], p[1], 0.0, 1.0],
                      'uv': [p[0] * 0.5 + 0.5, p[1] * 0.5 + 0.5]}

        self.check_mesh_data(ref, postvs)

        self.check_no_validation(f"{marker} debugging")

        rdtest.log.end_section(marker)

    def check_direct_dispatch(self):
        """Debug threads of the shader object dispatch, which reads a structured buffer, a constant buffer and a
        texture and writes to a structured buffer, a byte address buffer (atomically) and a storage image, all
        accessed directly from the heaps"""
        marker = "SO Direct Dispatch"
        rdtest.log.begin_section(marker)

        layout = self.get_layout("SO Direct Indices:")
        img = layout['imgSize']
        buf = layout['bufSize']
        samp = layout['sampSize']

        Image = rd.DescriptorType.Image
        RWImage = rd.DescriptorType.ReadWriteImage
        RWBuffer = rd.DescriptorType.ReadWriteBuffer
        CBuffer = rd.DescriptorType.ConstantBuffer
        Sampler = rd.DescriptorType.Sampler

        dispatch = self.find_action(marker).nextAction
        if not (dispatch.flags & rd.ActionFlags.Dispatch):
            raise rdtest.TestFailureException(f"Expected a dispatch after the marker, got {dispatch.customName}")

        self.controller.GetDebugMessages()
        self.set_event(dispatch.eventId, False)

        cs = rd.ShaderStage.Compute
        access = self.check_direct_accesses(marker, [
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

        self.check_no_validation("shader object dispatch feedback")

        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if not refl.debugInfo.debuggable:
            raise rdtest.TestFailureException(f"Shader object dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        resources = {r.name: r.resourceId for r in self.controller.GetResources()}

        # each thread writes (texture + inBuf[x % 2]) * scale + bias * x to outBuf[x]
        tex = [1.0, 1.0, 0.0, 1.0]
        in_buf = [[1.0, 0.5, 1.0, 1.0], [0.5, 0.5, 0.0, 0.0]]
        scale = [1.0, 1.0, 1.0, 0.5]
        bias = [0.25, 0.0, 0.5, 0.0]

        out_data = self.controller.GetBufferData(resources["directOutBuf"], 0, 64)

        # the counter is never reset, so it's incremented by 4 every frame
        counter = struct.unpack_from("I", self.controller.GetBufferData(resources["directCounterBuf"], 0, 4), 0)[0]
        if counter < 4 or (counter % 4) != 0:
            raise rdtest.TestFailureException(f"Dispatch counter is {counter}, expected a multiple of 4")

        # the first thread's output is (texture + buffer colour) * scale
        data = struct.unpack_from("4f", out_data, 0)
        if not rdtest.value_compare(list(data), [2.0, 1.5, 1.0, 1.0], eps=0.02):
            raise rdtest.TestFailureException(f"Shader object dispatch output is {list(data)}")

        rdtest.log.success("Shader object dispatch output is as expected")

        # the storage image's first pixel is the first thread's output too, clamped to the RGBA8 range
        written = list(self.controller.PickPixel(resources["directOutTex"], 0, 0, rd.Subresource(),
                                                 rd.CompType.Typeless).floatValue)
        if not rdtest.value_compare(written, [1.0, 1.0, 1.0, 1.0], eps=1.0 / 255.0):
            raise rdtest.TestFailureException(f"Shader object dispatch wrote {written} to the output image")

        dim = refl.dispatchThreadsDimension
        if list(dim) != [4, 1, 1]:
            raise rdtest.TestFailureException(f"Unexpected dispatch thread dimension {list(dim)}")

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

                prev = self.find_last_source_var(debug.trace, variables, 'prev').value.u32v[0]
                if prev < counter - 4 or prev >= counter:
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged 'prev' is {prev}, expected in [{counter - 4}, {counter})")

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

        self.check_no_validation(f"{marker} debugging")

        rdtest.log.end_section(marker)

    def check_partial_replay(self):
        """Replay into every action in reverse order, so the state restore has to rebind shader objects, heaps
        and push data each time, and the output of each draw is as expected"""
        expected = {}
        for marker, check in (("SO Direct Draw", (0.5, 0.5, [1.0, 0.5, 0.0, 1.0])),
                              ("SO Mapped Draw", (0.25, 0.25, [0.875, 0.25, 0.0, 1.0])),
                              ("SO Then Pipeline Draw", (0.5, 0.5, [0.5, 1.0, 0.0, 1.0])),
                              ("Pipeline Then SO Draw", (0.5, 0.5, [0.25, 0.75, 0.0, 1.0]))):
            expected[self.find_action(marker).nextAction.eventId] = check

        actions = [a for a in self.flatten_actions(self.controller.GetRootActions())
                   if a.flags & (rd.ActionFlags.Drawcall | rd.ActionFlags.Dispatch)]

        for a in reversed(actions):
            self.controller.SetFrameEvent(a.eventId, True)

            if a.eventId in expected:
                fx, fy, col = expected[a.eventId]
                v = self.controller.GetPipelineState().GetViewport(0)
                out = self.controller.GetPipelineState().GetOutputTargets()[0].resource
                self.check_pixel_value(out, int(v.x + v.width * fx), int(v.y + v.height * fy), col, eps=0.02)

        # now forwards, with partial replays between
        for a in actions:
            self.controller.SetFrameEvent(a.eventId, False)
            self.controller.GetPipelineState().GetShader(rd.ShaderStage.Vertex)

        rdtest.log.success(f"Partially replayed into {len(actions)} actions")

        self.check_no_validation("partial replays")
