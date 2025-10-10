import math

import torch
from torch import nn
from torch.nn import functional as F

from . import layers
from . import utils



def orthogonal_(module):
    nn.init.orthogonal_(module.weight)
    return module

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]



class ResLinearBlock(layers.ConditionedResidualBlock):
    def __init__(self, feats_in, c_in, c_mid, c_out, group_size=32, dropout_rate=0.):
        skip = None if c_in == c_out else orthogonal_(nn.Linear(c_in, c_out, bias=False))
        rnn = None if c_in == c_out else nn.LSTM(c_out, c_out, 1, batch_first=True)
        super().__init__(
            layers.AdaGN(feats_in, c_in, max(1, c_in // group_size)),
            nn.GELU(),
            nn.Linear(c_in, c_mid),
            nn.Dropout(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.GELU(),
            layers.TemporalAdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.Linear(c_mid, c_out),
            nn.Dropout(dropout_rate, inplace=True),
            skip=skip, rnn = rnn
            )


class DBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., downsample=False, self_attn=False, cross_attn=False, c_visual_enc=0, c_audio_enc=0):
        modules = [nn.Identity()]
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResLinearBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention1d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                if c_audio_enc >0:
                    modules.append(layers.CrossAttention1d(my_c_out, c_audio_enc, max(1, my_c_out // head_size), norm, dropout_rate, cond_key='speaker_audio'))
                if c_visual_enc >0:
                    modules.append(layers.CrossAttention1d(my_c_out, c_visual_enc, max(1, my_c_out // head_size), norm, dropout_rate, cond_key='speaker_3dmm'))
        super().__init__(*modules)


class UBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., upsample=False, self_attn=False, cross_attn=False, c_visual_enc=0, c_audio_enc=0):
        modules = []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResLinearBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention1d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                if c_audio_enc > 0:
                    modules.append(layers.CrossAttention1d(my_c_out, c_audio_enc, max(1, my_c_out // head_size), norm,
                                                           dropout_rate, cond_key='speaker_audio'))
                if c_visual_enc > 0:
                    modules.append(
                        layers.CrossAttention1d(my_c_out, c_visual_enc, max(1, my_c_out // head_size), norm, dropout_rate, cond_key='speaker_3dmm'))

        modules.append(nn.Identity())
        super().__init__(*modules)

    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = torch.cat([input, skip], dim=-1)
        return super().forward(input, cond)


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(orthogonal_(nn.Linear(feats_in if i == 0 else feats_out, feats_out)))
            layers.append(nn.GELU())
        super().__init__(*layers)


class ReactDiff(nn.Module):
    def __init__(self, c_in, feats_in, depths, channels, self_attn_depths, cross_attn_depths=None, mapping_cond_dim=0, unet_cond_dim=0, cross_visual_cond_dim=0, cross_audio_cond_dim=0, dropout_rate=0., skip_stages=0, has_variance=False):
        super().__init__()
        self.c_in = c_in
        self.channels = channels
        self.unet_cond_dim = unet_cond_dim
        self.has_variance = has_variance
        self.temporal_embed = layers.FourierFeatures(1, feats_in)
        self.timestep_embed = layers.TemporalFeatures(1, feats_in)

        self.cross_audio_cond_dim = cross_audio_cond_dim
        if mapping_cond_dim > 0:
            self.mapping_cond = nn.Linear(mapping_cond_dim, feats_in, bias=False)
        self.mapping = MappingNet(feats_in, feats_in)
        self.temporal_mapping = MappingNet(feats_in, feats_in)

        self.proj_in = nn.Linear(c_in + unet_cond_dim, channels[max(0, skip_stages - 1)])
        self.lstm_in = nn.LSTM(channels[max(0, skip_stages - 1)], channels[max(0, skip_stages - 1)], 1, batch_first=True)

        self.lstm_out = nn.LSTM(channels[max(0, skip_stages - 1)], channels[max(0, skip_stages - 1)], 1, batch_first=True)
        self.proj_out = nn.Linear(channels[max(0, skip_stages - 1)], c_in + (1 if self.has_variance else 0))
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        if cross_visual_cond_dim == 0 and cross_audio_cond_dim ==0:
            cross_attn_depths = [False] * len(self_attn_depths)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[max(0, i - 1)]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample=i > skip_stages, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_visual_enc=cross_visual_cond_dim, c_audio_enc=cross_audio_cond_dim, dropout_rate=dropout_rate))
        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[max(0, i - 1)]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample = False, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_visual_enc=cross_visual_cond_dim, c_audio_enc=cross_audio_cond_dim, dropout_rate=dropout_rate))

        self.u_net = layers.UNet(d_blocks, reversed(u_blocks), skip_stages=skip_stages)

    def forward(self, input, sigma, past_cond = None, mapping_cond=None, unet_cond=None, cross_cond=None, cross_cond_padding=None, return_variance=False, temporal_cond = None):
        c_noise = sigma.log() / 4

        timestep_embed = self.timestep_embed(append_dims(c_noise, 2))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        mapping_out = self.mapping(timestep_embed + mapping_cond_embed)
        cond = {'cond': mapping_out}

        if unet_cond is not None:
            input = torch.cat([input, unet_cond], dim=-1)
        if cross_cond is not None:
            cond['cross_padding'] = cross_cond_padding
            if cross_cond['speaker_3dmm'] is not None:
                cond['speaker_3dmm'] = cross_cond['speaker_3dmm']
            if cross_cond['speaker_audio'] is not None:
                cond['speaker_audio'] = cross_cond['speaker_audio']
        if temporal_cond is not None:
            temporal_cond = self.temporal_mapping(self.temporal_embed(append_dims(temporal_cond, 3)))
            cond['cond_temporal'] = temporal_cond

        input = self.proj_in(input)
        past_cond = self.proj_in(past_cond).transpose(0,1).contiguous()
        input, _ = self.lstm_in(input, (past_cond, torch.zeros_like(past_cond)))
        input = self.u_net(input, cond)
        input, _ = self.lstm_out(input, (past_cond, torch.zeros_like(past_cond)))
        input = self.proj_out(input)
        if self.has_variance:
            input, logvar = input[:, :-1], input[:, -1].flatten(1).mean(1)

        if self.has_variance and return_variance:
            return input, logvar
        return input

    def set_skip_stages(self, skip_stages):
        self.proj_in = nn.Linear(self.proj_in.in_channels, self.channels[max(0, skip_stages - 1)])
        self.proj_out = nn.Linear(self.channels[max(0, skip_stages - 1)], self.proj_out.out_channels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.u_net.skip_stages = skip_stages
        return self

    @torch.no_grad()
    def sampling(self, x, sigmas, past_cond, cond=None, temporal_cond=None, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
        """DPM-Solver++(2M) SDE."""

        if solver_type not in {'heun', 'midpoint'}:
            raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])

        old_denoised = None
        h_last = None

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = self.forward(x, sigmas[i] * s_in, past_cond, cross_cond=cond, temporal_cond=temporal_cond, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

                if eta:
                    x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h
        return x





if __name__ == '__main__':
    diffusion = ReactDiff(52, 128, [2, 4, 4], [128, 256, 512], [False, False, False])
    x = torch.randn(2, 28, 52)
    sigma = torch.randn(2, 1)
    y = diffusion(x, sigma)
    print(y.shape)