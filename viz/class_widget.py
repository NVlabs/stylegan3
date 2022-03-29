import imgui
from gui_utils import imgui_utils


# ----------------------------------------------------------------------------


class ClassWidget:
    def __init__(self, viz):
        self.viz = viz
        self.cls = 0
        self.animate = False
        self.count = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        cls = self.cls

        if show:
            imgui.text('Class')
            imgui.same_line(viz.label_w)
            with imgui_utils.grayed_out(not viz.result.get('is_conditional', False)):
                _changed, self.cls = imgui.slider_int('##cls', self.cls, 0, viz.result.get('num_classes', 0) - 1)
                imgui.same_line()
                _clicked, self.animate = imgui.checkbox('Anim##cls', self.animate)
                imgui.same_line()
                if imgui_utils.button('Reset', width=-1, enabled=(cls != self.cls or self.animate)):
                    self.cls = 0
                    self.animate = False
                    self.count = 0

            if self.animate:
                self.count += self.viz.frame_delta
                if self.count > 1.5:  # Update the class every 1.5 seconds; arbitrary, change as you will
                    self.cls = (self.cls + 1) % viz.result.get('num_classes')  # Loop back
                    self.count = 0

            # Sanity check when loading new networks
            self.cls = min(self.cls, viz.result.get('num_classes', 1) - 1)
            viz.args.update(cls=self.cls)
