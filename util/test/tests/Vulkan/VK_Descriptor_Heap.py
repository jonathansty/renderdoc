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
