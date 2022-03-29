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
from torch_utils import gen_utils

#----------------------------------------------------------------------------

class EquivarianceWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.xlate          = dnnlib.EasyDict(x=0, y=0, anim=False, round=False, speed=1e-2)
        self.xlate_def      = dnnlib.EasyDict(self.xlate)  # Save default value of translation above for reset button
        self.rotate         = dnnlib.EasyDict(val=0, anim=False, speed=5e-3)
        self.rotate_def     = dnnlib.EasyDict(self.rotate)  # Save default values of rotation above for reset button
        self.scale          = dnnlib.EasyDict(x=1, y=1, anim=False, round=False, speed=1e-1)
        self.scale_def      = dnnlib.EasyDict(self.scale)   # Save default values of scale above for reset button
        self.shear = dnnlib.EasyDict(x=0, y=0, anim=False, round=False, speed=1e-1)
        self.shear_def = dnnlib.EasyDict(self.shear)  # Save default values of shear above for reset button
        self.opts           = dnnlib.EasyDict(mirror_x=False, mirror_y=False, untransform=False)
        self.opts_def       = dnnlib.EasyDict(self.opts)  # Save default values of untransform above for reset button

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
            imgui.text('Scale')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, (self.scale.x, self.scale.y) = imgui.input_float2('##scale', self.scale.x, self.scale.y, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag fast##scale', width=viz.button_w)
            if dragging:
                self.scale.x += dx / viz.font_size * 2e-2
                self.scale.y += dy / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag slow##scale', width=viz.button_w)
            if dragging:
                self.scale.x += dx / viz.font_size * 4e-4
                self.scale.y += dy / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.scale.anim = imgui.checkbox('Anim##scale', self.scale.anim)
            imgui.same_line()
            _clicked, self.scale.round = imgui.checkbox('Round##scale', self.scale.round)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.scale.anim):
                changed, speed = imgui.slider_float('##scale_speed', self.scale.speed, 0, 0.5, format='Speed %.5f', power=5)
                if changed:
                    self.scale.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##scale', width=-1, enabled=(self.scale != self.scale_def)):
                self.scale = dnnlib.EasyDict(self.scale_def)

        if show:
            imgui.text('Shear')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, (self.shear.x, self.shear.y) = imgui.input_float2('##shear', self.shear.x, self.shear.y, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag fast##shear', width=viz.button_w)
            if dragging:
                self.shear.x += dx / viz.font_size * 2e-2
                self.shear.y += dy / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag slow##shear', width=viz.button_w)
            if dragging:
                self.shear.x += dx / viz.font_size * 4e-4
                self.shear.y += dy / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.shear.anim = imgui.checkbox('Anim##shear', self.shear.anim)
            imgui.same_line()
            _clicked, self.shear.round = imgui.checkbox('Round##shear', self.shear.round)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.shear.anim):
                changed, speed = imgui.slider_float('##shear_speed', self.shear.speed, 0, 0.5, format='Speed %.5f', power=5)
                if changed:
                    self.shear.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##shear', width=-1, enabled=(self.shear != self.shear_def)):
                self.shear = dnnlib.EasyDict(self.shear_def)

        if show:
            imgui.set_cursor_pos_x(imgui.get_content_region_max()[0] - 1 - viz.button_w*1 - viz.font_size*16.75)
            _clicked, self.opts.mirror_x = imgui.checkbox('Mirror x##opts', self.opts.mirror_x)
            imgui.same_line()
            _clicked, self.opts.mirror_y = imgui.checkbox('Mirror y##opts', self.opts.mirror_y)
            imgui.same_line()
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

        if self.scale.anim:
            c = np.array([self.scale.x, self.scale.y], dtype=np.float64)
            t = c.copy()
            if np.max(np.abs(t)) < 1e-4:
                t += 1
            t *= 0.1 / np.hypot(*t)
            t += c[::-1] * [1, -1]
            d = t - c
            d *= (viz.frame_delta * self.scale.speed) / np.hypot(*d)
            self.scale.x += d[0]
            self.scale.y += d[1]

        if self.shear.anim:
            c = np.array([self.shear.x, self.shear.y], dtype=np.float64)
            t = c.copy()
            if np.max(np.abs(t)) < 1e-4:
                t += 1
            t *= 0.1 / np.hypot(*t)
            t += c[::-1] * [1, -1]
            d = t - c
            d *= (viz.frame_delta * self.shear.speed) / np.hypot(*d)
            self.shear.x += d[0]
            self.shear.y += d[1]

        pos = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
        if self.xlate.round and 'img_resolution' in viz.result:
            pos = np.rint(pos * viz.result.img_resolution) / viz.result.img_resolution
        angle = self.rotate.val * np.pi * 2
        scale = np.array([self.scale.x, self.scale.y], dtype=np.float64)
        if self.scale.round and 'img_resolution' in viz.result:
            scale = np.rint(scale * viz.result.img_resolution) / viz.result.img_resolution
        shear = np.array([self.shear.x, self.shear.y], dtype=np.float64)
        if self.shear.round and 'img_resolution' in viz.result:
            shear = np.rint(shear * viz.result.img_resolution) / viz.result.img_resolution

        # Get the respective affine transform
        transform = gen_utils.make_affine_transform(angle=angle,                    # In terms of rev (1 rev = 2*pi rad)
                                                    translate_x=pos[0],             # In terms of G.img_resolution
                                                    translate_y=pos[1],             # In terms of G.img_resolution
                                                    scale_x=scale[0],               # If 0 < scale < 1, will zoom in
                                                    scale_y=scale[1],               # If 0 < scale < 1, will zoom in
                                                    shear_x=shear[0],               # Think of it as the slope of the vertical edges; best viewed if unstransform is ticked
                                                    shear_y=shear[1],               # Think of it as the slope of the horizontal edges; best viewed if unstransform is ticked
                                                    mirror_x=self.opts.mirror_x,
                                                    mirror_y=self.opts.mirror_y)

        viz.args.input_transform = transform.tolist()  # A list is expected

        viz.args.update(untransform=self.opts.untransform)

#----------------------------------------------------------------------------
