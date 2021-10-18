# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import copy
import traceback
import numpy as np
import torch
import torch.fft
import torch.nn
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

#----------------------------------------------------------------------------

def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

#----------------------------------------------------------------------------

def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self):
        self._device        = torch.device('cuda')
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        self._start_event   = torch.cuda.Event(enable_timing=True)
        self._end_event     = torch.cuda.Event(enable_timing=True)
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = dnnlib.EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).numpy()
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).numpy()
        if 'error' in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                net = copy.deepcopy(orig_net)
                net = self._tweak_network(net, **tweak_kwargs)
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _tweak_network(self, net):
        # Print diagnostics.
        #for name, value in misc.named_params_and_buffers(net):
        #    if name.endswith('.magnitude_ema'):
        #        value = value.rsqrt().numpy()
        #        print(f'{name:<50s}{np.min(value):<16g}{np.max(value):g}')
        #    if name.endswith('.weight') and value.ndim == 4:
        #        value = value.square().mean([1,2,3]).sqrt().numpy()
        #        print(f'{name:<50s}{np.min(value):<16g}{np.max(value):g}')
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def _render_impl(self, res,
        pkl             = None,
        w0_seeds        = [[0, 1]],
        stylemix_idx    = [],
        stylemix_seed   = 0,
        trunc_psi       = 1,
        trunc_cutoff    = 0,
        random_seed     = 0,
        noise_mode      = 'const',
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        fft_show        = False,
        fft_all         = True,
        fft_range_db    = 50,
        fft_beta        = 8,
        input_transform = None,
        untransform     = False,
    ):
        # Dig up network details.
        G = self.get_network(pkl, 'G_ema')
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        all_seeds = [seed for seed, _weight in w0_seeds] + [stylemix_seed]
        all_seeds = list(set(all_seeds))
        all_zs = np.zeros([len(all_seeds), G.z_dim], dtype=np.float32)
        all_cs = np.zeros([len(all_seeds), G.c_dim], dtype=np.float32)
        for idx, seed in enumerate(all_seeds):
            rnd = np.random.RandomState(seed)
            all_zs[idx] = rnd.randn(G.z_dim)
            if G.c_dim > 0:
                all_cs[idx, rnd.randint(G.c_dim)] = 1

        # Run mapping network.
        w_avg = G.mapping.w_avg
        all_zs = self.to_device(torch.from_numpy(all_zs))
        all_cs = self.to_device(torch.from_numpy(all_cs))
        all_ws = G.mapping(z=all_zs, c=all_cs, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
        all_ws = dict(zip(all_seeds, all_ws))

        # Calculate final W.
        w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
        stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < G.num_ws]
        if len(stylemix_idx) > 0:
            w[:, stylemix_idx] = all_ws[stylemix_seed][np.newaxis, stylemix_idx]
        w += w_avg

        # Run synthesis network.
        synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32)
        torch.manual_seed(random_seed)
        out, layers = self.run_synthesis_net(G.synthesis, w, capture_layer=layer_name, **synthesis_kwargs)

        # Update layer list.
        cache_key = (G.synthesis, tuple(sorted(synthesis_kwargs.items())))
        if cache_key not in self._net_layers:
            if layer_name is not None:
                torch.manual_seed(random_seed)
                _out, layers = self.run_synthesis_net(G.synthesis, w, **synthesis_kwargs)
            self._net_layers[cache_key] = layers
        res.layers = self._net_layers[cache_key]

        # Untransform.
        if untransform and res.has_input_transform:
            out, _mask = _apply_affine_transformation(out.to(torch.float32), G.synthesis.input.transform, amax=6) # Override amax to hit the fast path in upfirdn2d.

        # Select channels and compute statistics.
        out = out[0].to(torch.float32)
        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
        sel = out[base_channel : base_channel + sel_channels]
        res.stats = torch.stack([
            out.mean(), sel.mean(),
            out.std(), sel.std(),
            out.norm(float('inf')), sel.norm(float('inf')),
        ])

        # Scale and convert to uint8.
        img = sel
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img

        # FFT.
        if fft_show:
            sig = out if fft_all else sel
            sig = sig.to(torch.float32)
            sig = sig - sig.mean(dim=[1,2], keepdim=True)
            sig = sig * torch.kaiser_window(sig.shape[1], periodic=False, beta=fft_beta, device=self._device)[None, :, None]
            sig = sig * torch.kaiser_window(sig.shape[2], periodic=False, beta=fft_beta, device=self._device)[None, None, :]
            fft = torch.fft.fftn(sig, dim=[1,2]).abs().square().sum(dim=0)
            fft = fft.roll(shifts=[fft.shape[0] // 2, fft.shape[1] // 2], dims=[0,1])
            fft = (fft / fft.mean()).log10() * 10 # dB
            fft = self._apply_cmap((fft / fft_range_db + 1) / 2)
            res.image = torch.cat([img.expand_as(fft), fft], dim=1)

    @staticmethod
    def run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5: # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == '':
                    name = 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split('.')[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            out = net(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

#----------------------------------------------------------------------------
