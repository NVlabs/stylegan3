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

class LayerWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.prev_layers    = None
        self.cur_layer      = None
        self.sel_channels   = 3
        self.base_channel   = 0
        self.img_scale_db   = 0
        self.img_normalize  = False
        self.fft_show       = False
        self.fft_all        = True
        self.fft_range_db   = 50
        self.fft_beta       = 8
        self.refocus        = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        layers = viz.result.get('layers', [])
        if self.prev_layers != layers:
            self.prev_layers = layers
            self.refocus = True
        layer = ([layer for layer in layers if layer.name == self.cur_layer] + [None])[0]
        if layer is None and len(layers) > 0:
            layer = layers[-1]
            self.cur_layer = layer.name
        num_channels = layer.shape[1] if layer is not None else 0
        base_channel_max = max(num_channels - self.sel_channels, 0)

        if show:
            bg_color = [0.16, 0.29, 0.48, 0.2]
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # Begin list.
            width = viz.font_size * 28
            height = imgui.get_text_line_height_with_spacing() * 12 + viz.spacing
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *bg_color)
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin_child('##list', width=width, height=height, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

            # List items.
            for layer in layers:
                selected = (self.cur_layer == layer.name)
                _opened, selected = imgui.selectable(f'##{layer.name}_selectable', selected)
                imgui.same_line(viz.spacing)
                _clicked, selected = imgui.checkbox(f'{layer.name}##radio', selected)
                if selected:
                    self.cur_layer = layer.name
                    if self.refocus:
                        imgui.set_scroll_here()
                        viz.skip_frame() # Focus will change on next frame.
                        self.refocus = False
                imgui.same_line(width - viz.font_size * 13)
                imgui.text_colored('x'.join(str(x) for x in layer.shape[2:]), *dim_color)
                imgui.same_line(width - viz.font_size * 8)
                imgui.text_colored(str(layer.shape[1]), *dim_color)
                imgui.same_line(width - viz.font_size * 5)
                imgui.text_colored(layer.dtype, *dim_color)

            # End list.
            if len(layers) == 0:
                imgui.text_colored('No layers found', *dim_color)
            imgui.end_child()
            imgui.pop_style_color(4)
            imgui.pop_style_var(1)

            # Begin options.
            imgui.same_line()
            imgui.begin_child('##options', width=-1, height=height, border=False)

            # RGB & normalize.
            rgb = (self.sel_channels == 3)
            _clicked, rgb = imgui.checkbox('RGB', rgb)
            self.sel_channels = 3 if rgb else 1
            imgui.same_line(viz.font_size * 4)
            _clicked, self.img_normalize = imgui.checkbox('Normalize', self.img_normalize)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
            if imgui_utils.button('Reset##img_flags', width=-1, enabled=(self.sel_channels != 3 or self.img_normalize)):
                self.sel_channels = 3
                self.img_normalize = False

            # Image scale.
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                _changed, self.img_scale_db = imgui.slider_float('##scale', self.img_scale_db, min_value=-40, max_value=40, format='Scale %+.1f dB')
            imgui.same_line()
            if imgui_utils.button('Reset##scale', width=-1, enabled=(self.img_scale_db != 0)):
                self.img_scale_db = 0

            # Base channel.
            self.base_channel = min(max(self.base_channel, 0), base_channel_max)
            narrow_w = imgui.get_text_line_height_with_spacing()
            with imgui_utils.grayed_out(base_channel_max == 0):
                with imgui_utils.item_width(-1 - viz.button_w - narrow_w * 2 - viz.spacing * 3):
                    _changed, self.base_channel = imgui.drag_int('##channel', self.base_channel, change_speed=0.05, min_value=0, max_value=base_channel_max, format=f'Channel %d/{num_channels}')
                imgui.same_line()
                if imgui_utils.button('-##channel', width=narrow_w):
                    self.base_channel -= 1
                imgui.same_line()
                if imgui_utils.button('+##channel', width=narrow_w):
                    self.base_channel += 1
            imgui.same_line()
            self.base_channel = min(max(self.base_channel, 0), base_channel_max)
            if imgui_utils.button('Reset##channel', width=-1, enabled=(self.base_channel != 0 and base_channel_max > 0)):
                self.base_channel = 0

            # Stats.
            stats = viz.result.get('stats', None)
            stats = [f'{stats[idx]:g}' if stats is not None else 'N/A' for idx in range(6)]
            rows = [
                ['Statistic',   'All channels', 'Selected'],
                ['Mean',        stats[0],       stats[1]],
                ['Std',         stats[2],       stats[3]],
                ['Max',         stats[4],       stats[5]],
            ]
            height = imgui.get_text_line_height_with_spacing() * len(rows) + viz.spacing
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *bg_color)
            imgui.begin_child('##stats', width=-1, height=height, border=True)
            for y, cols in enumerate(rows):
                for x, col in enumerate(cols):
                    if x != 0:
                        imgui.same_line(viz.font_size * (4 + (x - 1) * 6))
                    if x == 0 or y == 0:
                        imgui.text_colored(col, *dim_color)
                    else:
                        imgui.text(col)
            imgui.end_child()
            imgui.pop_style_color(1)

            # FFT & all.
            _clicked, self.fft_show = imgui.checkbox('FFT', self.fft_show)
            imgui.same_line(viz.font_size * 4)
            with imgui_utils.grayed_out(not self.fft_show or base_channel_max == 0):
                _clicked, self.fft_all = imgui.checkbox('All channels', self.fft_all)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
            with imgui_utils.grayed_out(not self.fft_show):
                if imgui_utils.button('Reset##fft_flags', width=-1, enabled=(self.fft_show or not self.fft_all)):
                    self.fft_show = False
                    self.fft_all = True

            # FFT range.
            with imgui_utils.grayed_out(not self.fft_show):
                with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                    _changed, self.fft_range_db = imgui.slider_float('##fft_range_db', self.fft_range_db, min_value=0.1, max_value=100, format='Range +-%.1f dB')
                imgui.same_line()
                if imgui_utils.button('Reset##fft_range_db', width=-1, enabled=(self.fft_range_db != 50)):
                    self.fft_range_db = 50

            # FFT beta.
            with imgui_utils.grayed_out(not self.fft_show):
                with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                    _changed, self.fft_beta = imgui.slider_float('##fft_beta', self.fft_beta, min_value=0, max_value=50, format='Kaiser beta %.2f', power=2.63)
                imgui.same_line()
                if imgui_utils.button('Reset##fft_beta', width=-1, enabled=(self.fft_beta != 8)):
                    self.fft_beta = 8

            # End options.
            imgui.end_child()

        self.base_channel = min(max(self.base_channel, 0), base_channel_max)
        viz.args.layer_name = self.cur_layer if len(layers) > 0 and self.cur_layer != layers[-1].name else None
        viz.args.update(sel_channels=self.sel_channels, base_channel=self.base_channel, img_scale_db=self.img_scale_db, img_normalize=self.img_normalize)
        viz.args.fft_show = self.fft_show
        if self.fft_show:
            viz.args.update(fft_all=self.fft_all, fft_range_db=self.fft_range_db, fft_beta=self.fft_beta)

#----------------------------------------------------------------------------
