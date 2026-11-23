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
        # messages generated while loading the capture (e.g. by resource usage tracking) are only returned by the
        # first call, before any check consumes them
        self.check_no_validation("capture load")

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

        self.check_heap_disassembly()

        self.check_heap_descriptors()

        self.check_heap_view_variety()

        self.check_mapped_accesses()

        self.check_mapped_array_accesses()

        self.check_mapped_sources()

        self.check_direct_heap_access()

        self.check_direct_heap_dynamic()

        self.check_heap_texel_null()

        self.check_heap_buffers()

        self.check_heap_rebind()

        self.check_heap_mesh_output()

        self.check_heap_overlays()

        self.check_heap_pixel_history()

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

            # the shader debugger resolves the resources mapped from set/binding into the heaps (including the
            # embedded and custom border colour samplers), and must produce the same output as the real draw
            for fx, fy, _ in checks:
                self.check_debug_pixel(int(v.x + v.width * fx), int(v.y + v.height * fy))

            if test == 0:
                self.check_mapped_vertex_debug()

            self.check_no_validation(f"Heap Test {test} debugging")

            rdtest.log.end_section(f"Heap Test {test}")

    def check_mapped_vertex_debug(self):
        """Debug the vertices of the current draw (Heap Test 0), whose vertex shader has no resources but is
        a graphics pipeline using descriptor heaps"""
        positions = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]

        for vtx in range(4):
            with self.debug_vertex(vtx, 0, vtx, 0) as debug:
                cycles, variables = self.process_trace(debug.trace)

                pos_var = self.find_output_source_var(debug.trace, rd.ShaderBuiltin.Position)
                pos = list(self.evaluate_source_var(pos_var, variables).value.f32v[0:4])

                expected = [positions[vtx][0], positions[vtx][1], 0.0, 1.0]
                if not rdtest.value_compare(pos, expected, eps=1.0E-5):
                    raise rdtest.TestFailureException(f"Vertex {vtx} debugged position {pos}, expected {expected}")

                rdtest.log.success(f"Vertex {vtx} debugged in {cycles} cycles, position {pos} as expected")

    def heap_draws(self):
        """The heap graphics draws used by the overlay and pixel history checks, as (marker, action)"""
        draws = []
        for marker in ("Heap Test 0", "Direct Heap Draw", "Heap Buffer Draw"):
            action = self.find_action(marker)
            if action is None:
                raise rdtest.TestFailureException(f"Couldn't find marker {marker}")
            draws.append((marker, action.nextAction))
        return draws

    def check_heap_overlays(self):
        """Every debug overlay applicable to a draw, on draws using descriptor heaps"""
        out = self.controller.CreateOutput(rd.CreateHeadlessWindowingData(100, 100), rd.ReplayOutputType.Texture)

        for marker, draw in self.heap_draws():
            rdtest.log.begin_section(f"{marker} overlays")

            self.controller.GetDebugMessages()
            self.set_event(draw.eventId, True)

            pipe = self.controller.GetPipelineState()
            v = pipe.GetViewport(0)
            col_tex = pipe.GetOutputTargets()[0].resource

            # a point inside the draw (the middle of the viewport cell). The buffer draw is a triangle
            cx = int(v.x + v.width * 0.5)
            cy = int(v.y + v.height * (0.633 if marker == "Heap Buffer Draw" else 0.5))

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

    def check_heap_pixel_history(self):
        """Pixel history on draws using descriptor heaps"""
        for marker, draw in self.heap_draws():
            rdtest.log.begin_section(f"{marker} pixel history")

            self.controller.GetDebugMessages()
            self.set_event(draw.eventId, True)

            pipe = self.controller.GetPipelineState()
            v = pipe.GetViewport(0)
            col_tex = pipe.GetOutputTargets()[0].resource

            x = int(v.x + v.width * 0.5)
            y = int(v.y + v.height * (0.633 if marker == "Heap Buffer Draw" else 0.5))

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

    def check_heap_view_variety(self):
        """Image views with array/mip ranges, cube, depth and swizzled views, and buffer descriptors with ranges
        into the middle of a buffer"""
        view_action = self.find_action("Heap Views:")

        if view_action is None:
            raise rdtest.TestFailureException("Couldn't find the heap views marker")

        layout = {}
        for item in view_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        draw = self.find_action("Heap Test 0")
        self.set_event(draw.nextAction.eventId, False)

        res_heap = self.controller.GetVulkanPipelineState().resourceHeap.buffer

        img_size = layout['imgSize']
        buf_size = layout['bufSize']

        T = rd.TextureType
        S = rd.TextureSwizzle
        ident = (S.Red, S.Green, S.Blue, S.Alpha)

        # key, name, type, texture type, first mip, num mips, first slice, num slices, comp type, comp count, comp
        # byte width, swizzle
        images = [
            ('array', "viewArrayTex", rd.DescriptorType.Image, T.Texture2DArray, 0, 1, 1, 2,
             rd.CompType.UNorm, 4, 1, ident),
            ('mip', "viewMipTex", rd.DescriptorType.Image, T.Texture2D, 1, 2, 0, 1,
             rd.CompType.UNorm, 4, 1, ident),
            ('cube', "viewCubeTex", rd.DescriptorType.Image, T.TextureCube, 0, 1, 0, 6,
             rd.CompType.UNorm, 4, 1, ident),
            ('cubeStorage', "viewCubeTex", rd.DescriptorType.ReadWriteImage, T.Texture2DArray, 0, 1, 0, 6,
             rd.CompType.UNorm, 4, 1, ident),
            ('depth', "viewDepthTex", rd.DescriptorType.Image, T.Texture2D, 0, 1, 0, 1,
             rd.CompType.Depth, 1, 4, ident),
            ('swizzle', "viewFloatTex", rd.DescriptorType.Image, T.Texture2D, 0, 1, 0, 1,
             rd.CompType.Float, 4, 2, (S.Blue, S.Green, S.Red, S.Alpha)),
            ('swizzleR32', "viewR32Tex", rd.DescriptorType.Image, T.Texture2D, 0, 1, 0, 1,
             rd.CompType.Float, 1, 4, (S.One, S.Red, S.Zero, S.Alpha)),
        ]

        def fetch(offset, desc_type, size):
            r = rd.DescriptorRange()
            r.offset = offset
            r.descriptorSize = size
            r.count = 1
            r.type = desc_type
            return r

        def res_name(rid):
            return self.get_resource(rid).name if rid != rd.ResourceId() else None

        for key, name, typ, tex_type, first_mip, num_mips, first_slice, num_slices, comp, count, width, swz in images:
            r = fetch(layout[key] * img_size, typ, img_size)
            d = self.controller.GetDescriptors(res_heap, [r])[0]

            got = res_name(d.resource)
            if d.type != typ or got != name:
                raise rdtest.TestFailureException(f"View '{key}' expected to be {typ} {name}, got {d.type} {got}")

            if d.view == rd.ResourceId():
                raise rdtest.TestFailureException(f"View '{key}' has no view object")

            if d.textureType != tex_type:
                raise rdtest.TestFailureException(f"View '{key}' has texture type {d.textureType}, expected {tex_type}")

            got_range = (d.firstMip, d.numMips, d.firstSlice, d.numSlices)
            if got_range != (first_mip, num_mips, first_slice, num_slices):
                raise rdtest.TestFailureException(
                    f"View '{key}' has range {got_range}, expected {(first_mip, num_mips, first_slice, num_slices)}")

            f = d.format
            if f.compType != comp or f.compCount != count or f.compByteWidth != width:
                raise rdtest.TestFailureException(f"View '{key}' has unexpected format {f.Name()}")

            got_swz = (d.swizzle.red, d.swizzle.green, d.swizzle.blue, d.swizzle.alpha)
            if got_swz != swz:
                raise rdtest.TestFailureException(f"View '{key}' has swizzle {got_swz}, expected {swz}")

            loc = self.controller.GetDescriptorLocations(res_heap, [r])[0]
            if loc.logicalBindName != f"ResourceHeap[{layout[key]}]":
                raise rdtest.TestFailureException(f"View '{key}' has location name {loc.logicalBindName}")

        # the cube and its storage view are of the same image but different views
        cube_descs = self.controller.GetDescriptors(res_heap, [
            fetch(layout['cube'] * img_size, rd.DescriptorType.Image, img_size),
            fetch(layout['cubeStorage'] * img_size, rd.DescriptorType.ReadWriteImage, img_size)])
        if cube_descs[0].resource != cube_descs[1].resource or cube_descs[0].view == cube_descs[1].view:
            raise rdtest.TestFailureException("Cube and storage views should be different views of the same image")

        # buffers: a range in the middle of a bigger buffer, and a small range
        buffers = [
            ('ubo', rd.DescriptorType.ConstantBuffer, 0x40, 0x80),
            ('ssbo', rd.DescriptorType.ReadWriteBuffer, 0x40, 0x80),
            ('ssboPartial', rd.DescriptorType.ReadWriteBuffer, layout['partialOffset'], layout['partialSize']),
        ]

        for key, typ, offset, size in buffers:
            r = fetch(layout[key] * buf_size, typ, buf_size)
            d = self.controller.GetDescriptors(res_heap, [r])[0]

            got = res_name(d.resource)
            if d.type != typ or got != "viewRangeBuffer":
                raise rdtest.TestFailureException(f"Buffer '{key}' expected to be {typ} viewRangeBuffer, got {d.type} {got}")

            if d.byteOffset != offset or d.byteSize != size:
                raise rdtest.TestFailureException(
                    f"Buffer '{key}' has range {d.byteOffset}/{d.byteSize}, expected {offset}/{size}")

            loc = self.controller.GetDescriptorLocations(res_heap, [r])[0]
            if loc.logicalBindName != f"ResourceHeap[{layout[key]}]":
                raise rdtest.TestFailureException(f"Buffer '{key}' has location name {loc.logicalBindName}")

        rdtest.log.success("Heap view variety and buffer ranges decoded as expected")

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
            [(frag, rd.DescriptorType.Image, 'res', layout['tex'], 'tex0'),
             (frag, rd.DescriptorType.Sampler, 'embedded', 0, None)],
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
                if a.type == rd.DescriptorType.Sampler and not heap:
                    heap = ['embedded']
                if len(heap) != 1:
                    raise rdtest.TestFailureException(
                        f"Heap Test {test}: access {a.type} {a.index} has unexpected store {a.descriptorStore}")

                r = rd.DescriptorRange(a)

                if a.type == rd.DescriptorType.Sampler:
                    samp = self.controller.GetSamplerDescriptors(a.descriptorStore, [r])[0]
                    border = [round(x, 2) for x in samp.borderColorValue.floatValue]
                    if test == 7 and border != [0.5, 0.25, 0.75, 1.0]:
                        raise rdtest.TestFailureException(f"Heap Test {test}: sampler border is {border}")
                    if test == 6:
                        # embedded: nearest, clamp to opaque white border, no object, creation time constant
                        if (samp.filter.minify != rd.FilterMode.Point or samp.filter.magnify != rd.FilterMode.Point
                                or samp.addressU != rd.AddressMode.ClampBorder
                                or samp.addressV != rd.AddressMode.ClampBorder
                                or border != [1.0, 1.0, 1.0, 1.0] or samp.object != rd.ResourceId()
                                or not samp.creationTimeConstant):
                            raise rdtest.TestFailureException(f"Heap Test {test}: embedded sampler is wrong: {samp}")

                        loc = self.controller.GetDescriptorLocations(a.descriptorStore, [r])[0]
                        if loc.logicalBindName != "Embedded Sampler" or loc.category != rd.DescriptorCategory.Sampler:
                            raise rdtest.TestFailureException(
                                f"Heap Test {test}: embedded sampler location is {loc.logicalBindName}")
                    elif test != 7 and samp.addressU != rd.AddressMode.ClampEdge:
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

    def check_mapped_array_accesses(self):
        """Arrayed set/binding mappings are found by feedback, which reports the elements used. The draw reads
        textures 0, 1 and 3 of a 4 element image array (constant offset with an array stride) and element 1 of an
        SSBO array (push index base of 1 plus an array stride), so buffer 2 of the heap's array"""
        layout_action = self.find_action("Heap Array Layout:")

        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the heap array layout marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        draw = self.find_action("Heap Array Draw")

        if draw is None:
            raise rdtest.TestFailureException("Couldn't find the heap array draw marker")

        self.controller.GetDebugMessages()

        self.set_event(draw.nextAction.eventId, False)

        frag = rd.ShaderStage.Fragment
        ssbo_types = (rd.DescriptorType.Buffer, rd.DescriptorType.ReadWriteBuffer)

        pipe = self.controller.GetVulkanPipelineState()
        heaps = {'res': pipe.resourceHeap.buffer, 'samp': pipe.samplerHeap.buffer}

        refl = self.controller.GetPipelineState().GetShaderReflection(frag)

        # (type, array element, heap, byte offset, resource name, reflection name)
        expected = [
            (rd.DescriptorType.Image, 0, 'res', layout['texArr'], 'arrTex0', 'texArr'),
            (rd.DescriptorType.Image, 1, 'res', layout['texArr'] + layout['imgStride'], 'arrTex1', 'texArr'),
            (rd.DescriptorType.Image, 3, 'res', layout['texArr'] + layout['imgStride'] * 3, 'arrTex3', 'texArr'),
            (rd.DescriptorType.Sampler, 0, 'samp', layout['samp'], None, 'arrSamp'),
            ('ssbo', 1, 'res', layout['ssboArr'] + layout['bufStride'] * (layout['ssboBase'] + 1), 'arrSsbo2',
             'ssboArr'),
        ]

        got = []
        for a in self.controller.GetDescriptorAccess():
            if a.index == rd.DescriptorAccess.NoShaderBinding:
                continue

            # push constants (including the leftover compute pipeline's) are not heap mapped
            if a.type == rd.DescriptorType.ConstantBuffer:
                if not self.controller.GetPipelineState().GetShaderReflection(a.stage).constantBlocks[
                        a.index].bufferBacked:
                    continue
                raise rdtest.TestFailureException(f"Unexpected constant buffer access {a.index}")

            if a.stage != frag:
                raise rdtest.TestFailureException(f"Unexpected stage {a.stage} for access")

            if a.descriptorStore == rd.ResourceId():
                raise rdtest.TestFailureException(f"Access {a.type} {a.index}[{a.arrayElement}] has no store")

            heap = [k for k, v in heaps.items() if v == a.descriptorStore]
            if len(heap) != 1:
                raise rdtest.TestFailureException(
                    f"Access {a.type} {a.index}[{a.arrayElement}] has unexpected store {a.descriptorStore}")

            r = rd.DescriptorRange(a)

            if a.type == rd.DescriptorType.Sampler:
                res = None
                name = refl.samplers[a.index].name
                typ = a.type
            else:
                desc = self.controller.GetDescriptors(a.descriptorStore, [r])[0]
                res = self.get_resource(desc.resource).name if desc.resource != rd.ResourceId() else None
                typ = 'ssbo' if a.type in ssbo_types else a.type
                name = refl.readOnlyResources[a.index].name if a.type != rd.DescriptorType.ReadWriteBuffer else \
                    refl.readWriteResources[a.index].name

            # the logical location names the heap slot
            loc = self.controller.GetDescriptorLocations(a.descriptorStore, [r])[0]
            exp_loc = f"{'SamplerHeap' if heap[0] == 'samp' else 'ResourceHeap'}[{a.byteOffset // a.byteSize}]"
            if loc.logicalBindName != exp_loc:
                raise rdtest.TestFailureException(f"Access location is {loc.logicalBindName}, expected {exp_loc}")

            got.append((typ, a.arrayElement, heap[0], a.byteOffset, res, name))

        if sorted(got, key=str) != sorted(expected, key=str):
            raise rdtest.TestFailureException(
                f"Mapped array accesses don't match. Got: {sorted(got, key=str)} Expected: {sorted(expected, key=str)}")

        self.check_no_validation("mapped array access")

        rdtest.log.success(f"Heap Array Draw reported the expected {len(got)} mapped accesses")

        # tex0 + tex1 + tex3 + ssbo element: (0.5, 0, 0) + (0, 0.25, 0) + (0.25, 0.25, 0) + (0, 0, 0.5)
        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        self.check_pixel_value(out, int(v.x + v.width * 0.5), int(v.y + v.height * 0.5), (0.75, 0.5, 0.5, 1.0),
                               eps=0.02)

        # debugging resolves every array element used from the mappings: the indexing is dynamic in the shader
        self.check_debug_pixel(int(v.x + v.width * 0.5), int(v.y + v.height * 0.5))

        self.check_no_validation("mapped array draw")

        rdtest.log.success("Heap Array Draw output is as expected")

    def check_mapped_sources(self):
        """The other sources of set/binding mappings: push data, push address, resource heap data, indirect index,
        indirect index array, indirect address and combined image samplers (with constant offsets and a packed push
        index). Each is its own draw in the demo. Uniform buffers backed by push data are reported with the push
        constants store at an offset, buffers and combined image samplers which have no single descriptor in a heap
        are reported with a synthesised descriptor"""
        layout_action = self.find_action("Heap Source Layout:")

        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the heap source layout marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        img_stride = layout['imgStride']

        frag = rd.ShaderStage.Fragment
        cbuffer = rd.DescriptorType.ConstantBuffer
        image = rd.DescriptorType.Image
        sampler = rd.DescriptorType.Sampler
        combined = rd.DescriptorType.ImageSampler
        ssbo_types = (rd.DescriptorType.Buffer, rd.DescriptorType.ReadWriteBuffer)

        white = (1.0, 1.0, 1.0, 1.0)

        # each draw's output colours as (x fraction, y fraction, colour) within its viewport. The combined image
        # samplers scale the UVs by 2 so the upper right of the viewport is the sampler's white border colour
        #
        # also the expected accesses as (type, array element, where, byte offset, resource name), where 'where' is
        # the resource/sampler heap, or for the push constants and synthesised descriptors the descriptor location's
        # name. The offset of synthesised descriptors is just a table index so is not checked (None)
        draws = [
            ("Heap Source PushData", [(0.5, 0.5, (1.0, 0.5, 0.0, 1.0))],
             [(cbuffer, 0, "Push constants", layout['pushData'], 'push')]),
            ("Heap Source PushAddress", [(0.5, 0.5, (0.5, 0.0, 1.0, 1.0))],
             [(cbuffer, 0, "Push Address", None, 'pushAddrBuf')]),
            ("Heap Source HeapData", [(0.5, 0.5, (0.5, 1.0, 0.5, 1.0))],
             [(cbuffer, 0, "Heap Data", None, 'resHeap')]),
            ("Heap Source IndirectIndex", [(0.5, 0.5, (0.0, 0.5, 1.0, 1.0))],
             [(image, 0, 'res', layout['indTex'] + img_stride * layout['index'], 'indTex1'),
              (sampler, 0, 'samp', 0, None)]),
            # element 0 reads image 2 and element 1 image 0, the sum of both is the output
            ("Heap Source IndirectIndexArray", [(0.5, 0.5, (0.25, 0.25, 0.25, 1.0))],
             [(image, 0, 'res', layout['indTex'] + img_stride * layout['array0'], 'indTex2'),
              (image, 1, 'res', layout['indTex'] + img_stride * layout['array1'], 'indTex0'),
              (sampler, 0, 'samp', 0, None)]),
            ("Heap Source IndirectAddress", [(0.5, 0.5, (1.0, 0.0, 0.5, 1.0))],
             [('ssbo', 0, "Indirect Address", None, 'indAddrSsbo')]),
            ("Heap Source Combined", [(0.25, 0.25, (0.75, 0.75, 0.0, 1.0)), (0.75, 0.75, white)],
             [(combined, 0, "Combined Image Sampler", None, 'combTex')]),
            ("Heap Source CombinedPacked", [(0.25, 0.25, (0.0, 0.75, 0.75, 1.0)), (0.75, 0.75, white)],
             [(combined, 0, "Combined Image Sampler", None, 'packedTex1')]),
        ]

        eid = layout_action.eventId

        for name, checks, expected in draws:
            draw = self.find_action(name, eid)

            if draw is None:
                raise rdtest.TestFailureException(f"Couldn't find marker for {name}")

            rdtest.log.begin_section(name)

            eid = draw.eventId

            self.controller.GetDebugMessages()

            # selecting the event runs bindless feedback on the draw, which finds the array elements
            self.set_event(draw.nextAction.eventId, False)

            pipe = self.controller.GetVulkanPipelineState()
            heaps = {'res': pipe.resourceHeap.buffer, 'samp': pipe.samplerHeap.buffer}

            got = []
            details = []
            for a in self.controller.GetDescriptorAccess():
                if a.index == rd.DescriptorAccess.NoShaderBinding:
                    continue

                # push constants (including the leftover compute pipeline's) are not mapped
                if a.type == cbuffer and not self.controller.GetPipelineState().GetShaderReflection(
                        a.stage).constantBlocks[a.index].bufferBacked:
                    continue

                if a.stage != frag:
                    raise rdtest.TestFailureException(f"{name}: unexpected stage {a.stage} for access")

                if a.descriptorStore == rd.ResourceId():
                    raise rdtest.TestFailureException(f"{name}: access {a.type} {a.index} has no store")

                r = rd.DescriptorRange(a)

                where = [k for k, v in heaps.items() if v == a.descriptorStore]
                if where:
                    where = where[0]
                else:
                    where = self.controller.GetDescriptorLocations(a.descriptorStore, [r])[0].logicalBindName

                desc = None
                samp = None
                res = None

                if a.type == sampler:
                    samp = self.controller.GetSamplerDescriptors(a.descriptorStore, [r])[0]
                else:
                    desc = self.controller.GetDescriptors(a.descriptorStore, [r])[0]

                    if where == "Push constants":
                        res = 'push'
                    elif desc.resource != rd.ResourceId():
                        res = self.get_resource(desc.resource).name

                    if a.type == combined:
                        samp = self.controller.GetSamplerDescriptors(a.descriptorStore, [r])[0]

                typ = 'ssbo' if a.type in ssbo_types else a.type
                offset = a.byteOffset if where in ('res', 'samp', "Push constants") else None

                got.append((typ, a.arrayElement, where, offset, res))
                details.append((a, where, desc, samp))

            if sorted(got, key=str) != sorted(expected, key=str):
                raise rdtest.TestFailureException(
                    f"{name}: mapped accesses don't match. Got: {sorted(got, key=str)} "
                    f"Expected: {sorted(expected, key=str)}")

            rdtest.log.success(f"{name} reported the expected {len(got)} mapped accesses")

            # check the contents of the descriptors in more detail
            for a, where, desc, samp in details:
                if name == "Heap Source PushData":
                    # the push constants store is queried at an offset, and the viewed data is the rest of the push
                    # data from there
                    if not (desc.flags & rd.DescriptorFlags.InlineData) or desc.byteOffset != layout['pushData']:
                        raise rdtest.TestFailureException(f"{name}: unexpected descriptor {desc.flags} {desc.byteOffset}")
                    if desc.byteSize < 16:
                        raise rdtest.TestFailureException(f"{name}: unexpected descriptor size {desc.byteSize}")
                    self.check_source_buffer(name, desc, (1.0, 0.5, 0.0, 1.0))
                elif name == "Heap Source PushAddress":
                    # the buffer is at a non-zero offset, and the descriptor is the size the shader declares
                    if desc.type != cbuffer or desc.byteOffset != layout['pushAddressBuf'] or desc.byteSize != 16:
                        raise rdtest.TestFailureException(
                            f"{name}: unexpected descriptor {desc.type} {desc.byteOffset} {desc.byteSize}")
                    self.check_source_buffer(name, desc, (0.5, 0.0, 1.0, 1.0))
                elif name == "Heap Source HeapData":
                    # the data is in the resource heap buffer itself, at the heap offset plus the push data value
                    if (desc.type != cbuffer or desc.byteOffset != layout['heapData'] + layout['heapDataPush'] or
                            desc.byteSize != 16):
                        raise rdtest.TestFailureException(
                            f"{name}: unexpected descriptor {desc.type} {desc.byteOffset} {desc.byteSize}")
                    self.check_source_buffer(name, desc, (0.5, 1.0, 0.5, 1.0))
                elif name == "Heap Source IndirectAddress":
                    # the whole 0x100 byte buffer is visible
                    if desc.type not in ssbo_types or desc.byteOffset != 0 or desc.byteSize != 0x100:
                        raise rdtest.TestFailureException(
                            f"{name}: unexpected descriptor {desc.type} {desc.byteOffset} {desc.byteSize}")
                    self.check_source_buffer(name, desc, (1.0, 0.0, 0.5, 1.0))
                elif name in ("Heap Source Combined", "Heap Source CombinedPacked"):
                    # the sampler half is reported with the image half, and is the white border sampler in both. For
                    # the packed case sampler 0 would clamp to the edge instead
                    border = [round(x, 2) for x in samp.borderColorValue.floatValue]
                    if (desc.type != combined or samp.filter.minify != rd.FilterMode.Point or
                            samp.addressU != rd.AddressMode.ClampBorder or border != list(white)):
                        raise rdtest.TestFailureException(
                            f"{name}: unexpected combined descriptor {desc.type} {samp.filter.minify} "
                            f"{samp.addressU} {border}")
                elif name.startswith("Heap Source IndirectIndex"):
                    if a.type == sampler:
                        if samp.addressU != rd.AddressMode.ClampEdge:
                            raise rdtest.TestFailureException(f"{name}: sampler address is {samp.addressU}")

            self.check_no_validation(f"{name} mapped access")

            v = self.controller.GetPipelineState().GetViewport(0)
            out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

            for fx, fy, col in checks:
                self.check_pixel_value(out, int(v.x + v.width * fx), int(v.y + v.height * fy), col, eps=0.02)

            rdtest.log.success(f"{name} output is as expected")

            # the debugger resolves the sources itself (reading push data, the indirect memory and the buffers
            # at addresses) and must produce the same output
            for fx, fy, _ in checks:
                self.check_debug_pixel(int(v.x + v.width * fx), int(v.y + v.height * fy))

            self.check_no_validation(f"{name} debugging")

            rdtest.log.end_section(name)

    def check_source_buffer(self, name, desc, expected):
        """Checks that the first four floats of the buffer a descriptor views are as expected"""
        data = self.controller.GetBufferData(desc.resource, desc.byteOffset, 16)
        contents = list(struct.unpack_from("4f", data, 0))

        if not rdtest.value_compare(contents, list(expected), eps=1.0E-5):
            raise rdtest.TestFailureException(f"{name}: buffer contents are {contents}, expected {expected}")

    def check_heap_rebind(self):
        """Heap state changes within one command buffer: rebinding both heaps, only one of them,
        rebinding back, push data split over two pushes and a regular descriptor set pipeline"""
        layout_action = self.find_action("Heap2 Layout:")

        if layout_action is None:
            raise rdtest.TestFailureException("Couldn't find the second heap layout marker")

        layout = {}
        for item in layout_action.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        img = layout['imgSize']
        buf = layout['bufSize']
        samp = layout['sampDescSize']

        vs = rd.ShaderStage.Vertex
        ps = rd.ShaderStage.Pixel
        Image = rd.DescriptorType.Image
        RWBuffer = rd.DescriptorType.ReadWriteBuffer
        Sampler = rd.DescriptorType.Sampler

        # (reserved offset, reserved size, size) of each heap
        res1 = (layout['resReservedOffset1'], layout['resReservedSize1'], layout['resSize1'])
        samp1 = (layout['sampReservedOffset1'], layout['sampReservedSize1'], layout['sampSize1'])
        res2 = (layout['resReservedOffset'], layout['resReservedSize'], layout['resSize'])
        samp2 = (layout['sampReservedOffset'], layout['sampReservedSize'], layout['sampSize'])

        if res1[0] == res2[0] or res1[1] == res2[1] or samp1[0] == samp2[0] or samp1[1] == samp2[1]:
            raise rdtest.TestFailureException(
                f"Heaps should have different reserved ranges {res1} {res2} {samp1} {samp2}")

        grey = (0.2, 0.2, 0.2, 1.0)
        orange = (1.0, 0.5, 0.0, 1.0)
        azure = (0.0, 0.5, 1.0, 1.0)

        point_clamp = (rd.FilterMode.Point, rd.AddressMode.ClampEdge)
        linear_repeat = (rd.FilterMode.Linear, rd.AddressMode.Wrap)

        # the original heap's quad covers the middle half of the cell, the second heap's the middle 3/4
        # (name, resource heap, sampler heap, tex, buf, sampler, pixel checks)
        draws = [
            ("Heap Rebind Draw", ("resHeap2", res2), ("sampHeap2", samp2), "heap2Tex", "heap2Buf", linear_repeat,
             [(0.5, 0.5, azure), (0.15, 0.15, azure), (0.05, 0.05, grey)]),
            ("Heap Rebind One", ("resHeap", res1), ("sampHeap2", samp2), "directTex", "directBuf", linear_repeat,
             [(0.5, 0.5, orange), (0.15, 0.15, grey)]),
            ("Heap Rebind Back", ("resHeap", res1), ("sampHeap", samp1), "directTex", "directBuf", point_clamp,
             [(0.5, 0.5, orange), (0.15, 0.15, grey)]),
            ("Heap Push Offset Draw", ("resHeap", res1), ("sampHeap", samp1), "directTex", "directBuf",
             point_clamp, [(0.5, 0.5, orange), (0.15, 0.15, grey)]),
            ("Heap After Non-Heap", ("resHeap2", res2), ("sampHeap2", samp2), "heap2Tex", "heap2Buf",
             linear_repeat, [(0.5, 0.5, azure), (0.15, 0.15, azure), (0.05, 0.05, grey)]),
        ]

        def check_draw(name, res_heap, samp_heap, tex, bufname, sampler, pixels, debug_pixel=True):
            action = self.find_action(name)
            if action is None:
                raise rdtest.TestFailureException(f"Couldn't find marker {name}")

            draw = action.nextAction
            if not (draw.flags & rd.ActionFlags.Drawcall):
                raise rdtest.TestFailureException(f"Expected a draw after {name}, got {draw.customName}")

            self.controller.GetDebugMessages()
            self.set_event(draw.eventId, False)

            pipe = self.controller.GetVulkanPipelineState()

            for kind, heap, (expected_name, (reserved_off, reserved_size, size)) in (
                    ("resource", pipe.resourceHeap, res_heap), ("sampler", pipe.samplerHeap, samp_heap)):
                got_name = self.get_resource(heap.buffer).name if heap.buffer != rd.ResourceId() else None
                if got_name != expected_name:
                    raise rdtest.TestFailureException(f"{name}: {kind} heap is {got_name}, expected {expected_name}")

                if (heap.offset, heap.size, heap.reservedOffset, heap.reservedSize) != (0, size, reserved_off,
                                                                                       reserved_size):
                    raise rdtest.TestFailureException(
                        f"{name}: {kind} heap range {heap.offset}/{heap.size}/{heap.reservedOffset}/"
                        f"{heap.reservedSize}, expected 0/{size}/{reserved_off}/{reserved_size}")

            self.check_direct_accesses(name, [
                (vs, RWBuffer, 'res', layout['buf'] * buf, bufname),
                (ps, RWBuffer, 'res', layout['buf'] * buf, bufname),
                (ps, Image, 'res', layout['tex'] * img, tex),
                (ps, Sampler, 'samp', layout['samp'] * samp, None),
            ], samp_expect=sampler)

            v = self.controller.GetPipelineState().GetViewport(0)
            out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

            for fx, fy, col in pixels:
                self.check_pixel_value(out, int(v.x + v.width * fx), int(v.y + v.height * fy), col, eps=0.02)

            if debug_pixel:
                # the debugger must resolve the descriptors in the currently bound heaps
                self.check_debug_pixel(int(v.x + v.width * 0.5), int(v.y + v.height * 0.5))

            self.check_no_validation(f"{name} replay")

            rdtest.log.success(f"{name} is as expected")

        for d in draws:
            rdtest.log.begin_section(d[0])
            check_draw(*d)
            rdtest.log.end_section(d[0])

        # regular pipeline with a descriptor set between heap draws
        rdtest.log.begin_section("Non-Heap Draw")

        draw = self.find_action("Non-Heap Draw").nextAction

        self.controller.GetDebugMessages()
        self.set_event(draw.eventId, False)

        v = self.controller.GetPipelineState().GetViewport(0)
        out = self.controller.GetPipelineState().GetOutputTargets()[0].resource

        # the whole cell is covered
        purple = (0.5, 0.0, 1.0, 1.0)
        self.check_pixel_value(out, int(v.x + v.width * 0.5), int(v.y + v.height * 0.5), purple, eps=0.02)
        self.check_pixel_value(out, int(v.x + v.width * 0.1), int(v.y + v.height * 0.9), purple, eps=0.02)

        # the heap compute pipeline from an earlier dispatch is still bound, and its push constant access is reported
        # regardless of bind point (as for any stale compute pipeline), so ignore compute stage accesses
        access = [a for a in self.controller.GetDescriptorAccess() if a.stage != rd.ShaderStage.Compute]
        if len(access) != 1 or access[0].index == rd.DescriptorAccess.NoShaderBinding or access[0].stage != ps \
                or access[0].type != rd.DescriptorType.ImageSampler:
            raise rdtest.TestFailureException(
                f"Unexpected descriptor access for the non-heap draw: {[(a.stage, a.type, a.index) for a in access]}")

        pipe = self.controller.GetVulkanPipelineState()
        if access[0].descriptorStore in (pipe.resourceHeap.buffer, pipe.samplerHeap.buffer):
            raise rdtest.TestFailureException("Non-heap access should not be in a heap")

        desc = self.controller.GetDescriptors(access[0].descriptorStore, [rd.DescriptorRange(access[0])])[0]
        if self.get_resource(desc.resource).name != "nonHeapTex":
            raise rdtest.TestFailureException(
                f"Non-heap descriptor is {self.get_resource(desc.resource).name}, expected nonHeapTex")

        self.check_no_validation("Non-Heap Draw replay")

        rdtest.log.end_section("Non-Heap Draw")

        # partial replay: jump around in the frame, from the end back into the middle of the heap state
        # changes, and forwards again from earlier in the frame
        rdtest.log.begin_section("Heap rebind partial replay")

        last = self.get_last_action()
        for name, *rest in reversed(draws):
            self.set_event(last.eventId, True)
            check_draw(name, *rest, debug_pixel=False)

        for name, *rest in draws:
            self.set_event(self.find_action("Heap Test 0").nextAction.eventId, True)
            check_draw(name, *rest, debug_pixel=False)

        rdtest.log.end_section("Heap rebind partial replay")

    def check_heap_disassembly(self):
        """Every shader used by a draw or dispatch must disassemble. The Slang-compiled direct heap shaders put
        non-semantic debug instructions between block terminators and labels, which the disassembler must skip"""
        count = 0
        for action in self.controller.GetRootActions():
            for a in [action] + self.flatten_actions(action.children):
                if not (a.flags & (rd.ActionFlags.Drawcall | rd.ActionFlags.Dispatch)):
                    continue

                self.controller.SetFrameEvent(a.eventId, False)
                pipe = self.controller.GetPipelineState()

                compute = bool(a.flags & rd.ActionFlags.Dispatch)
                stages = [rd.ShaderStage.Compute] if compute else [rd.ShaderStage.Vertex, rd.ShaderStage.Pixel]
                pipe_obj = pipe.GetComputePipelineObject() if compute else pipe.GetGraphicsPipelineObject()

                for stage in stages:
                    refl = pipe.GetShaderReflection(stage)
                    if refl is None:
                        continue
                    disasm = self.controller.DisassembleShader(pipe_obj, refl, "")
                    if disasm == "" or "error" in disasm.lower()[:200]:
                        raise rdtest.TestFailureException(
                            f"Disassembly of {stage} at event {a.eventId} failed: {disasm[:200]}")
                    count += 1

        rdtest.log.success(f"Disassembled {count} shaders")

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

    def check_direct_heap_dynamic(self):
        """Dispatch with heap indices computed at runtime: each thread loads the descriptor at
        base + min(tid, count - 1) and a loop reads base..base+count-1. Only count of the consecutive
        images may be reported as accessed, and the debugger must read the right descriptor for each
        dynamic index."""
        marker = self.find_action("Direct Heap Dynamic Indices:")
        if marker is None:
            raise rdtest.TestFailureException("Couldn't find the direct heap dynamic indices marker")

        layout = {}
        for item in marker.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        img = layout['imgSize']
        buf = layout['bufSize']
        samp = layout['sampSize']
        base = layout['base']
        count = layout['count']

        Image = rd.DescriptorType.Image
        RWBuffer = rd.DescriptorType.ReadWriteBuffer
        Sampler = rd.DescriptorType.Sampler

        rdtest.log.begin_section("Direct Heap Dynamic Dispatch")

        dispatch = self.find_action("Direct Heap Dynamic Dispatch").nextAction
        if not (dispatch.flags & rd.ActionFlags.Dispatch):
            raise rdtest.TestFailureException(f"Expected a dispatch after the marker, got {dispatch.customName}")

        self.controller.GetDebugMessages()
        self.set_event(dispatch.eventId, False)

        # only the first 'count' of the 4 consecutive images are accessed, the last one is not
        cs = rd.ShaderStage.Compute
        expected_access = [(cs, Image, 'res', (base + i) * img, f"heapDynTex{i}") for i in range(count)]
        expected_access.append((cs, RWBuffer, 'res', layout['outBuf'] * buf, "heapDynOut"))
        expected_access.append((cs, Sampler, 'samp', layout['samp'] * samp, None))
        self.check_direct_accesses("Dynamic dispatch", expected_access)

        self.check_no_validation("direct heap dynamic dispatch feedback")

        pipe = self.controller.GetVulkanPipelineState()
        res_heap = pipe.resourceHeap.buffer
        samp_heap = pipe.samplerHeap.buffer

        resources = {r.name: r.resourceId for r in self.controller.GetResources()}

        # solid colours of the textures, count of 3 means 0..2 are accessed
        cols = [[0.2, 0.0, 0.0, 1.0], [0.0, 0.4, 0.0, 1.0], [0.0, 0.0, 0.6, 1.0], [0.8, 0.8, 0.0, 1.0]]

        total = [sum(cols[i][c] for i in range(count)) for c in range(4)]

        out_data = self.controller.GetBufferData(resources["heapDynOut"], 0, 128)

        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if not refl.debugInfo.debuggable:
            raise rdtest.TestFailureException(
                f"Dynamic direct heap dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        for x in range(4):
            expected_direct = cols[min(x, count - 1)]

            written = list(struct.unpack_from("4f", out_data, x * 16))
            if not rdtest.value_compare(written, expected_direct, eps=0.01):
                raise rdtest.TestFailureException(f"Thread {x} wrote {written}, expected {expected_direct}")

            written_sum = list(struct.unpack_from("4f", out_data, 64 + x * 16))
            if not rdtest.value_compare(written_sum, total, eps=0.01):
                raise rdtest.TestFailureException(f"Thread {x} wrote sum {written_sum}, expected {total}")

        rdtest.log.success("Dynamic dispatch output buffer has the expected values")

        # thread 0 and thread 2 index different descriptors. Thread 3 is clamped to the same as thread 2
        for x in (0, 2, 3):
            with self.debug_thread((0, 0, 0), (x, 0, 0)) as debug:
                cycles, variables = self.process_trace(debug.trace)

                expected_direct = cols[min(x, count - 1)]

                idx = self.find_last_source_var(debug.trace, variables, 'idx').value.u32v[0]
                if idx != base + min(x, count - 1):
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged 'idx' is {idx}, expected {base + min(x, count - 1)}")

                direct = list(self.find_last_source_var(debug.trace, variables, 'direct').value.f32v[0:4])
                if not rdtest.value_compare(direct, expected_direct, eps=0.01):
                    raise rdtest.TestFailureException(
                        f"Thread {x} debugged 'direct' is {direct}, expected {expected_direct}")

                total_var = list(self.find_last_source_var(debug.trace, variables, 'sum').value.f32v[0:4])
                if not rdtest.value_compare(total_var, total, eps=0.01):
                    raise rdtest.TestFailureException(f"Thread {x} debugged 'sum' is {total_var}, expected {total}")

                # the handle variables only hold the last descriptors loaded: the thread's own index, and the
                # last element the loop read. Neither may be the unused one
                dbg = {d for d in self.get_direct_heap_vars(variables, res_heap, samp_heap) if d[1] == Image}
                own = (rd.VarType.ReadOnlyResource, Image, 'res', (base + min(x, count - 1)) * img)
                last = (rd.VarType.ReadOnlyResource, Image, 'res', (base + count - 1) * img)
                allowed = {(rd.VarType.ReadOnlyResource, Image, 'res', (base + i) * img) for i in range(count)}
                if own not in dbg or last not in dbg or not dbg.issubset(allowed):
                    raise rdtest.TestFailureException(
                        f"Thread {x} image heap variables {sorted(dbg, key=str)} should include {own} and "
                        f"{last} and be within {sorted(allowed, key=str)}")

                rdtest.log.success(f"Thread {x} debugged in {cycles} cycles, 'direct' = {direct} as expected")

        self.check_no_validation("direct heap dynamic dispatch debugging")

    def check_heap_texel_null(self):
        """Uniform/storage texel buffer descriptors with sub-ranges at non-zero offsets, accessed by a direct heap
        dispatch (including debugging it), and null descriptors of each type if the device supports them"""
        marker = self.find_action("Heap Texel Indices:")
        if marker is None:
            raise rdtest.TestFailureException("Couldn't find the heap texel indices marker")

        layout = {}
        for item in marker.customName.split(":")[1].split():
            k, v = item.split("=")
            layout[k] = int(v)

        img = layout['imgSize']
        in_off = layout['in'] * img
        out_off = layout['out'] * img

        TB = rd.DescriptorType.TypedBuffer
        RWTB = rd.DescriptorType.ReadWriteTypedBuffer

        rdtest.log.begin_section("Heap Texel Dispatch")

        dispatch = self.find_action("Heap Texel Dispatch").nextAction
        if not (dispatch.flags & rd.ActionFlags.Dispatch):
            raise rdtest.TestFailureException(f"Expected a dispatch after the marker, got {dispatch.customName}")

        self.controller.GetDebugMessages()
        self.set_event(dispatch.eventId, False)

        cs = rd.ShaderStage.Compute
        self.check_direct_accesses("Texel dispatch", [
            (cs, TB, 'res', in_off, "heapTexelIn"),
            (cs, RWTB, 'res', out_off, "heapTexelOut"),
        ])

        self.check_no_validation("texel dispatch feedback")

        res_heap = self.controller.GetVulkanPipelineState().resourceHeap.buffer

        def fetch(offset, desc_type):
            r = rd.DescriptorRange()
            r.offset = offset
            r.descriptorSize = img
            r.count = 1
            r.type = desc_type
            return r

        # descriptors decode as the right type, buffer, format and sub-range
        for name, off, typ, buf_name, range_off, range_size in [
            ("input", in_off, TB, "heapTexelIn", layout['inOffset'], layout['inSize']),
            ("output", out_off, RWTB, "heapTexelOut", layout['outOffset'], layout['outSize'])]:
            d = self.controller.GetDescriptors(res_heap, [fetch(off, typ)])[0]

            got = self.get_resource(d.resource).name if d.resource != rd.ResourceId() else None
            if d.type != typ or got != buf_name:
                raise rdtest.TestFailureException(f"Texel {name} expected {typ} {buf_name}, got {d.type} {got}")

            if d.byteOffset != range_off or d.byteSize != range_size:
                raise rdtest.TestFailureException(
                    f"Texel {name} has range {d.byteOffset}/{d.byteSize}, expected {range_off}/{range_size}")

            f = d.format
            if f.compType != rd.CompType.Float or f.compCount != 4 or f.compByteWidth != 4:
                raise rdtest.TestFailureException(f"Texel {name} has unexpected format {f.Name()}")

            loc = self.controller.GetDescriptorLocations(res_heap, [fetch(off, typ)])[0]
            if loc.logicalBindName != f"ResourceHeap[{layout['in'] if name == 'input' else layout['out']}]":
                raise rdtest.TestFailureException(f"Texel {name} has location name {loc.logicalBindName}")

        rdtest.log.success("Texel buffer descriptors decoded as expected")

        # the output buffer: texel i of the input sub-range (global texel 16 + i) transformed
        resources = {r.name: r.resourceId for r in self.controller.GetResources()}

        def texel(i):
            return [float(i), float(i) * 0.5, 100.0 + float(i), float(i) * 0.25]

        def expected_out(x):
            v = texel(16 + x)
            return [v[0] * 2.0, v[1] * 2.0, v[2] * 2.0, v[3] + float(x)]

        out_data = self.controller.GetBufferData(resources["heapTexelOut"], layout['outOffset'], 16 * layout['texels'])

        for x in range(layout['texels']):
            written = list(struct.unpack_from("4f", out_data, x * 16))
            if not rdtest.value_compare(written, expected_out(x), eps=0.001):
                raise rdtest.TestFailureException(f"Texel {x} wrote {written}, expected {expected_out(x)}")

        rdtest.log.success("Texel output buffer has the expected values")

        refl = self.controller.GetPipelineState().GetShaderReflection(cs)
        if not refl.debugInfo.debuggable:
            raise rdtest.TestFailureException(f"Texel dispatch is not debuggable: {refl.debugInfo.debugStatus}")

        for x in (0, 1, 3):
            with self.debug_thread((0, 0, 0), (x, 0, 0)) as debug:
                cycles, variables = self.process_trace(debug.trace)

                v = list(self.find_last_source_var(debug.trace, variables, 'v').value.f32v[0:4])
                if not rdtest.value_compare(v, texel(16 + x), eps=0.001):
                    raise rdtest.TestFailureException(f"Thread {x} debugged 'v' is {v}, expected {texel(16 + x)}")

                rdtest.log.success(f"Thread {x} debugged in {cycles} cycles, read texel {v} as expected")

        self.check_no_validation("texel dispatch debugging")

        rdtest.log.end_section("Heap Texel Dispatch")

        # null descriptors
        null_marker = self.find_action("Heap Null:")
        if null_marker is None:
            rdtest.log.print("No null descriptors in this capture (nullDescriptor not supported), skipping")
            return

        null_layout = {}
        for item in null_marker.customName.split(":")[1].split():
            k, v = item.split("=")
            null_layout[k] = int(v)

        size = null_layout['size']

        D = rd.DescriptorType
        nulls = [
            ('sampledImage', D.Image),
            ('storageImage', D.ReadWriteImage),
            ('ubo', D.ConstantBuffer),
            ('ssbo', D.ReadWriteBuffer),
            ('texelBuffer', D.TypedBuffer),
            ('storageTexelBuffer', D.ReadWriteTypedBuffer),
        ]

        for key, typ in nulls:
            r = rd.DescriptorRange()
            r.offset = null_layout[key] * size
            r.descriptorSize = size
            r.count = 1
            r.type = typ

            d = self.controller.GetDescriptors(res_heap, [r])[0]

            if d.resource != rd.ResourceId() or d.type != typ:
                raise rdtest.TestFailureException(
                    f"Null {key} descriptor should be empty {typ}, got {d.type} {d.resource}")

        rdtest.log.success("Null descriptors decoded as empty")

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
