import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.func import update_registered_buffers, get_scale_table
from utils.ckbd import *
from modules.transform import *


class ELIC(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch # [8, 8, 8, 8, 16, 16, 32, 32, 96, 96]
        self.quant = config.quant # noise or ste
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEX(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEX(N, M, act=nn.ReLU)
        # Hyper Transform
        self.h_a = HyperAnalysisEX(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEX(N, M, act=nn.ReLU)
        # Channel Fusion Model
        self.local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quant == 'ste':
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset
        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == 'ste':
                    slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = self.gaussian_conditional.quantize(
                        slice_anchor, "noise" if self.training else "dequantize"
                    )
                    slice_anchor = ckbd_anchor(slice_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == 'ste':
                    slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = self.gaussian_conditional.quantize(
                        slice_nonanchor, "noise" if self.training else "dequantize"
                    )
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == 'ste':
                    slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = self.gaussian_conditional.quantize(
                        slice_anchor, "noise" if self.training else "dequantize"
                    )
                    slice_anchor = ckbd_anchor(slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == 'ste':
                    slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = self.gaussian_conditional.quantize(
                        slice_nonanchor, "noise" if self.training else "dequantize"
                    )
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False
        x_hat = self.g_s(y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
