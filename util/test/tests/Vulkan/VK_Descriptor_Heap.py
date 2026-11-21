import renderdoc as rd
import rdtest


class VK_Descriptor_Heap(rdtest.TestCase):
    demos_test_name = 'VK_Descriptor_Heap'

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

        eid = 0

        for test, checks in enumerate(expected):
            draw = self.find_action(f"Heap Test {test}", eid)

            if draw is None:
                raise rdtest.TestFailureException(f"Couldn't find marker for Heap Test {test}")

            rdtest.log.begin_section(f"Heap Test {test}")

            eid = draw.eventId

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

        # names of the heap locations
        locs = self.controller.GetDescriptorLocations(res_heap, [ranges[3]])
        n = layout['texArray'] // img_stride
        if [l.logicalBindName for l in locs] != [f"ResourceHeap[{n}]", f"ResourceHeap[{n + 1}]"]:
            raise rdtest.TestFailureException(f"Unexpected resource heap location names {[l.logicalBindName for l in locs]}")

        locs = self.controller.GetDescriptorLocations(samp_heap, [samp_ranges[1]])
        if locs[0].logicalBindName != f"SamplerHeap[{layout['customSamp'] // samp_stride}]":
            raise rdtest.TestFailureException(f"Unexpected sampler heap location name {locs[0].logicalBindName}")

        rdtest.log.success("Heap descriptors decoded as expected")
