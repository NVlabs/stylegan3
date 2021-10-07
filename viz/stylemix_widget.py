# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imgui
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class StyleMixingWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.seed_def   = 1000
        self.seed       = self.seed_def
        self.animate    = False
        self.enables    = []

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        num_enables = viz.result.get('num_ws', 18)
        self.enables += [False] * max(num_enables - len(self.enables), 0)

        if show:
            imgui.text('Stylemix')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8), imgui_utils.grayed_out(num_ws == 0):
                _changed, self.seed = imgui.input_int('##seed', self.seed)
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            with imgui_utils.grayed_out(num_ws == 0):
                _clicked, self.animate = imgui.checkbox('Anim', self.animate)

            pos2 = imgui.get_content_region_max()[0] - 1 - viz.button_w
            pos1 = pos2 - imgui.get_text_line_height() - viz.spacing
            pos0 = viz.label_w + viz.font_size * 12
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            for idx in range(num_enables):
                imgui.same_line(round(pos0 + (pos1 - pos0) * (idx / (num_enables - 1))))
                if idx == 0:
                    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3)
                with imgui_utils.grayed_out(num_ws == 0):
                    _clicked, self.enables[idx] = imgui.checkbox(f'##{idx}', self.enables[idx])
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f'{idx}')
            imgui.pop_style_var(1)

            imgui.same_line(pos2)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - 3)
            with imgui_utils.grayed_out(num_ws == 0):
                if imgui_utils.button('Reset', width=-1, enabled=(self.seed != self.seed_def or self.animate or any(self.enables[:num_enables]))):
                    self.seed = self.seed_def
                    self.animate = False
                    self.enables = [False] * num_enables

        if any(self.enables[:num_ws]):
            viz.args.stylemix_idx = [idx for idx, enable in enumerate(self.enables) if enable]
            viz.args.stylemix_seed = self.seed & ((1 << 32) - 1)
        if self.animate:
            self.seed += 1

#----------------------------------------------------------------------------
