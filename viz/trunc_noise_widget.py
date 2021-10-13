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

class TruncationNoiseWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.prev_num_ws    = 0
        self.trunc_psi      = 1
        self.trunc_cutoff   = 0
        self.noise_enable   = True
        self.noise_seed     = 0
        self.noise_anim     = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        has_noise = viz.result.get('has_noise', False)
        if num_ws > 0 and num_ws != self.prev_num_ws:
            if self.trunc_cutoff > num_ws or self.trunc_cutoff == self.prev_num_ws:
                self.trunc_cutoff = num_ws
            self.prev_num_ws = num_ws

        if show:
            imgui.text('Truncate')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10), imgui_utils.grayed_out(num_ws == 0):
                _changed, self.trunc_psi = imgui.slider_float('##psi', self.trunc_psi, -1, 2, format='Psi %.2f')
            imgui.same_line()
            if num_ws == 0:
                imgui_utils.button('Cutoff 0', width=(viz.font_size * 8 + viz.spacing), enabled=False)
            else:
                with imgui_utils.item_width(viz.font_size * 8 + viz.spacing):
                    changed, new_cutoff = imgui.slider_int('##cutoff', self.trunc_cutoff, 0, num_ws, format='Cutoff %d')
                    if changed:
                        self.trunc_cutoff = min(max(new_cutoff, 0), num_ws)

            with imgui_utils.grayed_out(not has_noise):
                imgui.same_line()
                _clicked, self.noise_enable = imgui.checkbox('Noise##enable', self.noise_enable)
                imgui.same_line(round(viz.font_size * 27.7))
                with imgui_utils.grayed_out(not self.noise_enable):
                    with imgui_utils.item_width(-1 - viz.button_w - viz.spacing - viz.font_size * 4):
                        _changed, self.noise_seed = imgui.input_int('##seed', self.noise_seed)
                    imgui.same_line(spacing=0)
                    _clicked, self.noise_anim = imgui.checkbox('Anim##noise', self.noise_anim)

            is_def_trunc = (self.trunc_psi == 1 and self.trunc_cutoff == num_ws)
            is_def_noise = (self.noise_enable and self.noise_seed == 0 and not self.noise_anim)
            with imgui_utils.grayed_out(is_def_trunc and not has_noise):
                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                if imgui_utils.button('Reset', width=-1, enabled=(not is_def_trunc or not is_def_noise)):
                    self.prev_num_ws = num_ws
                    self.trunc_psi = 1
                    self.trunc_cutoff = num_ws
                    self.noise_enable = True
                    self.noise_seed = 0
                    self.noise_anim = False

        if self.noise_anim:
            self.noise_seed += 1
        viz.args.update(trunc_psi=self.trunc_psi, trunc_cutoff=self.trunc_cutoff, random_seed=self.noise_seed)
        viz.args.noise_mode = ('none' if not self.noise_enable else 'const' if self.noise_seed == 0 else 'random')

#----------------------------------------------------------------------------
