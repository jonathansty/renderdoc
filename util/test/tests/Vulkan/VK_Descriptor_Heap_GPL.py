import struct
import renderdoc as rd
import rdtest


class VK_Descriptor_Heap_GPL(rdtest.TestCase):
    demos_test_name = 'VK_Descriptor_Heap_GPL'

    # marker, direct, ssbo index, ssbo name, quad scale, centre sample, pixel checks (fx, fy, colour)
    draws = [
        ("GPL Direct Draw", True, 0, None, 0.5, None, [(0.5, 0.5, [1.0, 0.5, 0.0, 1.0]),
                                                      (0.1, 0.1, [0.2, 0.2, 0.2, 1.0])]),
        ("GPL Mapped Draw", False, 1, "ssbo1", 1.0, None, [(0.25, 0.25, [0.875, 0.25, 0.0, 1.0]),
                                                           (0.75, 0.75, [0.875, 0.75, 0.5, 1.0])]),
        ("GPL Mapped Draw Optimized", False, 1, "ssbo1", 1.0, None,
         [(0.25, 0.25, [0.875, 0.25, 0.0, 1.0]), (0.75, 0.75, [0.875, 0.75, 0.5, 1.0])]),
        # the quad covers the middle half of the viewport, the uv still covers 0-1 across it
        ("GPL Shared Fragment Draw", False, 0, "ssbo0", 0.5, None,
         [(0.35, 0.35, [0.75, 0.125, 0.125, 1.0]), (0.65, 0.65, [0.75, 0.625, 0.625, 1.0]),
          (0.1, 0.1, [0.2, 0.2, 0.2, 1.0])]),
    ]

    def get_replay_options(self):
        opts = rdtest.TestCase.get_replay_options(self)
        # patched pipelines/shaders used by feedback must not generate validation errors
        opts.apiValidation = True
        return opts

    def check_capture(self):
        # messages generated while loading the capture are only returned by the first call, before any check
        # consumes them
        self.check_no_validation("capture load")

        self.check_pipeline_state()

        self.check_disassembly()

        self.get_layout("GPL Layout:")
        self.get_layout("GPL Direct Indices:")

        for marker, direct, ssbo_idx, ssbo_name, scale, _, checks in self.draws:
            if direct:
                self.check_direct_draw(marker, checks)
            else:
                self.check_mapped_draw(marker, ssbo_idx, ssbo_name, scale, checks)

        self.check_overlays()

        self.check_pixel_history()

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

    def check_pipeline_state(self):
        """Every draw reports the heaps and a linked pipeline, with both shaders resolved from the libraries"""
        pipes = {}
        for marker, *_ in self.draws:
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

            if pipe.graphics.pipelineResourceId == rd.ResourceId():
                raise rdtest.TestFailureException(f"{marker}: missing graphics pipeline")

            for stage in (pipe.vertexShader, pipe.fragmentShader):
                if stage.resourceId == rd.ResourceId():
                    raise rdtest.TestFailureException(f"{marker}: missing {stage.stage} shader")
                if stage.shaderObject:
                    raise rdtest.TestFailureException(f"{marker}: {stage.stage} is unexpectedly a shader object")

            pipes[marker] = self.get_resource(pipe.graphics.pipelineResourceId).name

            rdtest.log.success(f"{marker}: heaps and linked pipeline {pipes[marker]} as expected")

        expected = {"GPL Direct Draw": "directPipe", "GPL Mapped Draw": "mappedPipe",
                    "GPL Mapped Draw Optimized": "optPipe", "GPL Shared Fragment Draw": "sharedPipe"}
        if pipes != expected:
            raise rdtest.TestFailureException(f"Unexpected pipelines bound {pipes}, expected {expected}")

        self.check_no_validation("pipeline state")

    def check_disassembly(self):
        """Every shader used by a draw must disassemble"""
        count = 0
        for action in self.controller.GetRootActions():
            for a in [action] + self.flatten_actions(action.children):
                if not (a.flags & rd.ActionFlags.Drawcall):
                    continue

                self.controller.SetFrameEvent(a.eventId, False)
                pipe = self.controller.GetPipelineState()

                pipe_obj = pipe.GetGraphicsPipelineObject()

                for stage in (rd.ShaderStage.Vertex, rd.ShaderStage.Pixel):
                    refl = pipe.GetShaderReflection(stage)
                    if refl is None:
                        raise rdtest.TestFailureException(f"No reflection for {stage} at event {a.eventId}")

                    disasm = self.controller.DisassembleShader(pipe_obj, refl, "")
                    if disasm == "" or "error" in disasm.lower()[:200]:
                        raise rdtest.TestFailureException(
                            f"Disassembly of {stage} at event {a.eventId} failed: {disasm[:200]}")
                    count += 1

        if count != 8:
            raise rdtest.TestFailureException(f"Expected to disassemble 8 shaders, got {count}")

        rdtest.log.success(f"Disassembled {count} shaders")

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

    def check_outputs(self, marker, draw, checks):
        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        for fx, fy, col in checks:
            x = int(v.x + v.width * fx)
            y = int(v.y + v.height * fy)

            self.check_pixel_value(out, x, y, col, eps=0.02)

        rdtest.log.success(f"{marker} output is as expected")

    def check_direct_draw(self, marker, checks):
        """A direct heap draw through a pipeline linked from libraries, which reads a structured buffer from the
        heap in both stages and samples a texture in the pixel shader"""
        rdtest.log.begin_section(marker)

        layout = self.get_layout("GPL Direct Indices:")
        buf = layout['bufSize']
        img = layout['imgSize']
        samp = layout['sampSize']

        draw = self.find_action(marker).nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        vs = rd.ShaderStage.Vertex
        ps = rd.ShaderStage.Pixel
        RWBuffer = rd.DescriptorType.ReadWriteBuffer

        self.check_direct_accesses(marker, [
            (vs, RWBuffer, 'res', layout['buf'] * buf, "directBuf"),
            (ps, RWBuffer, 'res', layout['buf'] * buf, "directBuf"),
            (ps, rd.DescriptorType.Image, 'res', layout['tex'] * img, "directTex"),
            (ps, rd.DescriptorType.Sampler, 'samp', layout['samp'] * samp, None),
        ])

        self.check_no_validation(f"{marker} feedback")

        self.check_outputs(marker, draw, checks)

        v = self.controller.GetPipelineState().GetViewport(0)

        # vertex output
        postvs = self.get_postvs(draw, rd.MeshDataStage.VSOut)
        if not postvs:
            raise rdtest.TestFailureException(f"No VS Out data for {marker}")

        ref = {}
        for i, p in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            ref[i] = {'vtx': i, 'idx': i, '_Position': [p[0] * 0.5, p[1] * 0.5, 0.0, 1.0],
                      'entryPointParam_vsmain.uv': [p[0] * 0.5 + 0.5, p[1] * 0.5 + 0.5]}

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
                expected_direct = (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['buf'] * buf)
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
                    (rd.VarType.ReadWriteResource, RWBuffer, 'res', layout['buf'] * buf),
                    (rd.VarType.ReadOnlyResource, rd.DescriptorType.Image, 'res', layout['tex'] * img),
                    (rd.VarType.Sampler, rd.DescriptorType.Sampler, 'samp', layout['samp'] * samp),
                }
                if not expected_direct.issubset(direct):
                    raise rdtest.TestFailureException(
                        f"{marker} pixel {x},{y} heap descriptor variables {sorted(direct, key=str)} don't "
                        f"include {sorted(expected_direct, key=str)}")

        self.check_no_validation(f"{marker} debugging")

        rdtest.log.end_section(marker)

    def check_mapped_draw(self, marker, ssbo_idx, ssbo_name, scale, checks):
        """Draw with a pipeline linked from libraries, where the set/binding mappings (including a push index and
        an embedded sampler) were only chained on the stages in the libraries"""
        rdtest.log.begin_section(marker)

        layout = self.get_layout("GPL Layout:")
        buf_stride = layout['bufStride']

        draw = self.find_action(marker).nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        pipe = self.controller.GetVulkanPipelineState()

        heaps = {'res': pipe.resourceHeap.buffer, 'samp': pipe.samplerHeap.buffer}
        ssbo_types = (rd.DescriptorType.Buffer, rd.DescriptorType.ReadWriteBuffer)

        frag = rd.ShaderStage.Fragment

        # (stage, type, heap, byte offset, resource name). The push index selects the SSBO
        expected = [
            (frag, rd.DescriptorType.Image, 'res', layout['tex'], 'tex0'),
            (frag, rd.DescriptorType.Sampler, 'samp', layout['samp'], None),
            (frag, rd.DescriptorType.Sampler, 'embedded', 0, None),
            (frag, rd.DescriptorType.ConstantBuffer, 'res', layout['ubo'], 'ubo'),
            (frag, 'ssbo', 'res', layout['ssbo'] + buf_stride * ssbo_idx, ssbo_name),
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

        self.check_outputs(marker, draw, checks)

        v = self.controller.GetPipelineState().GetViewport(0)

        for fx, fy, col in checks:
            if col == [0.2, 0.2, 0.2, 1.0]:
                continue

            x = int(v.x + v.width * fx)
            y = int(v.y + v.height * fy)

            self.check_debug_pixel(x, y)

        # the vertex shader has no resources but is part of a pipeline using descriptor heaps
        positions = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]

        for vtx in range(4):
            with self.debug_vertex(vtx, 0, vtx, 0) as debug:
                cycles, variables = self.process_trace(debug.trace)

                pos_var = self.find_output_source_var(debug.trace, rd.ShaderBuiltin.Position)
                pos = list(self.evaluate_source_var(pos_var, variables).value.f32v[0:4])

                expected_pos = [positions[vtx][0] * scale, positions[vtx][1] * scale, 0.0, 1.0]
                if not rdtest.value_compare(pos, expected_pos, eps=1.0E-5):
                    raise rdtest.TestFailureException(f"Vertex {vtx} debugged position {pos}, expected {expected_pos}")

        # VS Out
        postvs = self.get_postvs(draw, rd.MeshDataStage.VSOut)
        if not postvs:
            raise rdtest.TestFailureException(f"No VS Out data for {marker}")

        ref = {}
        for i, p in enumerate(positions):
            ref[i] = {'vtx': i, 'idx': i, 'gl_Position': [p[0] * scale, p[1] * scale, 0.0, 1.0],
                      'uv': [p[0] * 0.5 + 0.5, p[1] * 0.5 + 0.5]}

        self.check_mesh_data(ref, postvs)

        self.check_no_validation(f"{marker} debugging")

        rdtest.log.end_section(marker)

    def check_overlays(self):
        """Every debug overlay applicable to a draw, which recreates the linked pipelines"""
        out = self.controller.CreateOutput(rd.CreateHeadlessWindowingData(100, 100), rd.ReplayOutputType.Texture)

        for marker, *_ in self.draws:
            draw = self.find_action(marker).nextAction

            rdtest.log.begin_section(f"{marker} overlays")

            self.controller.GetDebugMessages()
            self.set_event(draw.eventId, True)

            pipe = self.controller.GetPipelineState()
            v = pipe.GetViewport(0)
            col_tex = pipe.GetOutputTargets()[0].resource

            # a point inside the draw (the middle of the viewport cell, avoiding the shared fragment draw's
            # border sampling edge)
            cx = int(v.x + v.width * 0.45)
            cy = int(v.y + v.height * 0.45)

            tex = rd.TextureDisplay()
            tex.resourceId = col_tex

            for overlay in rd.DebugOverlay:
                if overlay == rd.DebugOverlay.NoOverlay:
                    continue

                # These overlays are just displaymodes really, not actually separate overlays
                if overlay == rd.DebugOverlay.NaN or overlay == rd.DebugOverlay.Clipping:
                    continue

                tex.overlay = overlay
                out.SetTextureDisplay(tex)
                out.Display()

                overlay_id = out.GetDebugOverlayTexID()

                picked = list(self.pick_pixel(overlay_id, cx, cy, rd.Subresource(), rd.CompType.Typeless).floatValue)

                rdtest.log.print(f"{marker}: overlay {overlay!s} centre pixel {picked}")

                eps = 1.0 / 256.0

                if overlay == rd.DebugOverlay.Drawcall:
                    # draw pixels are highlighted
                    self.check_pixel_value(overlay_id, cx, cy, [0.8, 0.1, 0.8, 1.0], eps=eps)
                elif overlay == rd.DebugOverlay.Wireframe:
                    # wireframe only covers the edges: a row through the draw must cross some wireframe
                    # pixels but not be filled with them
                    hits = 0
                    for px in range(int(v.x), int(v.x + v.width)):
                        p = self.pick_pixel(overlay_id, px, cy, rd.Subresource(), rd.CompType.Typeless)
                        if p.floatValue[3] != 0.0:
                            hits += 1
                    if hits == 0 or hits > v.width // 4:
                        raise rdtest.TestFailureException(
                            f"{marker}: wireframe overlay has {hits} wireframe pixels along the row")
                elif overlay in (rd.DebugOverlay.Depth, rd.DebugOverlay.BackfaceCull):
                    # no depth test and nothing is culled, so the draw passes everywhere it draws
                    self.check_pixel_value(overlay_id, cx, cy, [0.0, 1.0, 0.0, 1.0], eps=eps)
                elif overlay == rd.DebugOverlay.ViewportScissor:
                    if picked[3] == 0.0:
                        raise rdtest.TestFailureException(f"{marker}: viewport overlay is blank inside the viewport")
                elif overlay in (rd.DebugOverlay.QuadOverdrawDraw, rd.DebugOverlay.QuadOverdrawPass):
                    # refused for heap pipelines (TODO(heap slice 4)): the overlay is left blank rather than crashing
                    if picked != [0.0, 0.0, 0.0, 0.0]:
                        raise rdtest.TestFailureException(
                            f"{marker}: quad overdraw should be blank with descriptor heaps, got {picked}")
                elif overlay in (rd.DebugOverlay.TriangleSizeDraw, rd.DebugOverlay.TriangleSizePass):
                    if picked[3] == 0.0:
                        raise rdtest.TestFailureException(f"{marker}: triangle size overlay is blank inside the draw")

                self.check_no_validation(f"{marker} overlay {overlay!s}")

            rdtest.log.end_section(f"{marker} overlays")

        out.Shutdown()

    def check_pixel_history(self):
        """Pixel history on draws using linked heap pipelines"""
        for marker, *_ in self.draws:
            draw = self.find_action(marker).nextAction

            rdtest.log.begin_section(f"{marker} pixel history")

            self.controller.GetDebugMessages()
            self.set_event(draw.eventId, True)

            pipe = self.controller.GetPipelineState()
            v = pipe.GetViewport(0)
            col_tex = pipe.GetOutputTargets()[0].resource

            x = int(v.x + v.width * 0.45)
            y = int(v.y + v.height * 0.45)

            picked = list(self.pick_pixel(col_tex, x, y, rd.Subresource(), rd.CompType.Typeless).floatValue)

            history = self.controller.PixelHistory(col_tex, x, y, rd.Subresource(), rd.CompType.Typeless)

            mods = [m for m in history if m.eventId == draw.eventId]
            if len(mods) == 0:
                raise rdtest.TestFailureException(
                    f"{marker}: pixel history at {x},{y} has no modification for event {draw.eventId}: "
                    f"{[m.eventId for m in history]}")

            for m in mods:
                if not m.Passed():
                    raise rdtest.TestFailureException(f"{marker}: modification {m.eventId} did not pass")

            m = mods[-1]

            shader_out = list(m.shaderOut.col.floatValue)
            post = list(m.postMod.col.floatValue)

            rdtest.log.print(f"{marker}: picked {picked} shaderOut {shader_out} postMod {post}")

            if not rdtest.value_compare(post, picked, eps=0.01):
                raise rdtest.TestFailureException(f"{marker}: postMod {post} doesn't match pixel {picked}")

            if not rdtest.value_compare(shader_out, picked, eps=0.01):
                raise rdtest.TestFailureException(f"{marker}: shaderOut {shader_out} doesn't match pixel {picked}")

            self.check_no_validation(f"{marker} pixel history")

            rdtest.log.end_section(f"{marker} pixel history")

    def check_partial_replay(self):
        """Replay into every action in reverse order, so the state restore has to rebind the pipelines, heaps and
        push data each time, and the output of each draw is as expected"""
        expected = {}
        for marker, _, _, _, _, _, checks in self.draws:
            expected[self.find_action(marker).nextAction.eventId] = checks[0]

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
