# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class EquivarianceWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.xlate          = dnnlib.EasyDict(x=0, y=0, anim=False, round=False, speed=1e-2)
        self.xlate_def      = dnnlib.EasyDict(self.xlate)
        self.rotate         = dnnlib.EasyDict(val=0, anim=False, speed=5e-3)
        self.rotate_def     = dnnlib.EasyDict(self.rotate)
        self.opts           = dnnlib.EasyDict(untransform=False)
        self.opts_def       = dnnlib.EasyDict(self.opts)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Translate')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, (self.xlate.x, self.xlate.y) = imgui.input_float2('##xlate', self.xlate.x, self.xlate.y, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag fast##xlate', width=viz.button_w)
            if dragging:
                self.xlate.x += dx / viz.font_size * 2e-2
                self.xlate.y += dy / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag slow##xlate', width=viz.button_w)
            if dragging:
                self.xlate.x += dx / viz.font_size * 4e-4
                self.xlate.y += dy / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.xlate.anim = imgui.checkbox('Anim##xlate', self.xlate.anim)
            imgui.same_line()
            _clicked, self.xlate.round = imgui.checkbox('Round##xlate', self.xlate.round)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.xlate.anim):
                changed, speed = imgui.slider_float('##xlate_speed', self.xlate.speed, 0, 0.5, format='Speed %.5f', power=5)
                if changed:
                    self.xlate.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##xlate', width=-1, enabled=(self.xlate != self.xlate_def)):
                self.xlate = dnnlib.EasyDict(self.xlate_def)

        if show:
            imgui.text('Rotate')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, self.rotate.val = imgui.input_float('##rotate', self.rotate.val, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag fast##rotate', width=viz.button_w)
            if dragging:
                self.rotate.val += dx / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag slow##rotate', width=viz.button_w)
            if dragging:
                self.rotate.val += dx / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.rotate.anim = imgui.checkbox('Anim##rotate', self.rotate.anim)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.rotate.anim):
                changed, speed = imgui.slider_float('##rotate_speed', self.rotate.speed, -1, 1, format='Speed %.4f', power=3)
                if changed:
                    self.rotate.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##rotate', width=-1, enabled=(self.rotate != self.rotate_def)):
                self.rotate = dnnlib.EasyDict(self.rotate_def)

        if show:
            imgui.set_cursor_pos_x(imgui.get_content_region_max()[0] - 1 - viz.button_w*1 - viz.font_size*16)
            _clicked, self.opts.untransform = imgui.checkbox('Untransform', self.opts.untransform)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
            if imgui_utils.button('Reset##opts', width=-1, enabled=(self.opts != self.opts_def)):
                self.opts = dnnlib.EasyDict(self.opts_def)

        if self.xlate.anim:
            c = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
            t = c.copy()
            if np.max(np.abs(t)) < 1e-4:
                t += 1
            t *= 0.1 / np.hypot(*t)
            t += c[::-1] * [1, -1]
            d = t - c
            d *= (viz.frame_delta * self.xlate.speed) / np.hypot(*d)
            self.xlate.x += d[0]
            self.xlate.y += d[1]

        if self.rotate.anim:
            self.rotate.val += viz.frame_delta * self.rotate.speed

        pos = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
        if self.xlate.round and 'img_resolution' in viz.result:
            pos = np.rint(pos * viz.result.img_resolution) / viz.result.img_resolution
        angle = self.rotate.val * np.pi * 2

        viz.args.input_transform = [
            [np.cos(angle),  np.sin(angle), pos[0]],
            [-np.sin(angle), np.cos(angle), pos[1]],
            [0, 0, 1]]

        viz.args.update(untransform=self.opts.untransform)

#----------------------------------------------------------------------------
