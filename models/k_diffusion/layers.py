import math
import random
from einops import rearrange, repeat
import torch
from torch import nn
from torch.nn import functional as F
import random
from . import sampling, utils
# Karras et al. preconditioned denoiser

symmetry_actions = [
    (0, 1),
    (3, 4),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
    (20, 21),
    (23, 25),
    (27, 28),
    (29, 30),
    (32, 38),
    (33, 34),
    (35, 36),
    (43, 44),
    (45, 46),
    (47, 48),
    (49, 50)
]

co_occurred_actions = [
    (3, 16),
    (4, 17),
    (10, 0),
    (11, 1),
    (8, 0),
    (9, 1),
    (20, 3),
    (21, 4),
    (20, 2),
    (21, 2),
    (6, 32),
    (7, 38),
    (6, 43),
    (7, 44),
    (6, 29),
    (7, 30),
    (6, 27),
    (7, 28),
    (6, 47),
    (7, 48),
    (6, 35),
    (7, 36),
    (3, 43),
    (4, 44),
    (49, 6),
    (50, 7),
    (29, 0),
    (30, 1),
    (47, 0),
    (48, 1)
]

exclusive_actions = [
    (0, 3),
    (1, 4),
    (0, 16),
    (1, 17),
    (0, 20),
    (1, 21),
    (2, 8),
    (2, 9),
    (10, 16),
    (11, 17),
    (12, 14),
    (13, 15),
    (12, 18),
    (13, 19),
    (20, 8),
    (21, 9),
    (24, 26),
    (26, 47),
    (26, 48),
    (26, 51),
    (26, 33),
    (26, 34),
    (29, 43),
    (30, 44),
    (29, 47),
    (30, 48),
    (29, 31),
    (30, 31),
    (29, 30),
    (31, 33),
    (31, 34),
    (31, 39),
    (31, 43),
    (31, 44),
    (31, 51),
    (33, 35),
    (34, 36),
    (33, 47),
    (34, 48),
    (33, 43),
    (34, 44),
    (35, 45),
    (36, 46),
    (35, 47),
    (36, 48),
    (37, 24),
    (31, 24),
    (37, 47),
    (37, 48),
    (37, 33),
    (37, 34),
    (39, 40),
    (41, 42),
    (41, 51),
    (43, 45),
    (44, 46),
    (43, 47),
    (44, 48)
]


class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1., weighting='karras', weight_kinematics_loss=0.01, weight_velocity_loss=1):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.sml1 = nn.SmoothL1Loss(reduce=False)
        self.weight_kinematics_loss = weight_kinematics_loss
        self.weight_velocity_loss = weight_velocity_loss


        if callable(weighting):
            self.weighting = weighting
        if weighting == 'karras':
            self.weighting = torch.ones_like
        elif weighting == 'soft-min-snr':
            self.weighting = self._weighting_soft_min_snr
        else:
            raise ValueError(f'Unknown weighting type {weighting}')

    def _weighting_soft_min_snr(self, sigma):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** 2) ** 2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss_ori(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1) * c_weight


    def loss(self, input, noise, sigma, window_size, cross_cond, mask, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)

        frame_num = input.shape[1]
        interval_num = frame_num // window_size
        model_output = None
        past_frame = torch.zeros((input.shape[0], window_size, input.shape[2])).to(input.get_device())

        for i in range(0, interval_num):
            current_cond_dict = {}
            if i != (interval_num - 1):
                current_noise_listener_in = past_frame
                current_cond_dict['speaker_audio'] = cross_cond['speaker_audio'][:, 2 * i * window_size: 2 * (i + 1) * window_size]
                current_cond_dict['speaker_3dmm'] = cross_cond['speaker_3dmm'][:, i * window_size: (i + 1) * window_size]
            else:
                current_cond_dict['speaker_audio'] = cross_cond['speaker_audio'][:, 2 * i * window_size:]
                current_cond_dict['speaker_3dmm'] = cross_cond['speaker_3dmm'][:, i * window_size:]
                interval = current_cond_dict['speaker_audio'].shape[1] // 2
                current_noise_listener_in = c_out * model_output[:, -interval:, :] + c_skip * noised_input[:, -interval:, :]

            temporal_cond = torch.arange(i * window_size,  i * window_size + current_noise_listener_in.shape[1])
            temporal_cond = temporal_cond.view(1,-1,1).repeat(input.shape[0], 1, 1).view(input.shape[0], -1, 1).to(input.get_device())

            current_model_output = self.inner_model(current_noise_listener_in * c_in, sigma, past_frame[:, -1, :].unsqueeze(1), cross_cond = current_cond_dict, temporal_cond = temporal_cond, **kwargs)

            sign = random.randint(0, 1)
            if sign == 0:
                past_frame = input[:, i * window_size: (i + 1) * window_size, :]
            else:
                past_frame = c_out * current_model_output[:, :, :] + c_skip * noised_input[:,  i * window_size: (i + 1) * window_size, :]

            if i != 0:
                model_output = torch.cat((model_output, current_model_output), 1)
            else:
                model_output = current_model_output

        target = (input - c_skip * noised_input) / c_out
        kinematics_loss = self.kinematics_loss(model_output, target, mask.view(-1, 1, 1))
        velocity_loss = self.velocity_loss(model_output, target, window_size, mask.view(-1, 1, 1))
        loss = (model_output - target).pow(2).flatten(1).mean(1) * c_weight + self.weight_velocity_loss * velocity_loss + self.weight_kinematics_loss * kinematics_loss
        return loss


    def forward(self, input, sigma, past_cond, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        out = self.inner_model(input * c_in, sigma, past_cond, **kwargs)
        return out * c_out + input * c_skip

    def forward_online(self, input, sigma, window_size, **kwargs):

        frame_num = input.shape[1]
        interval_num = frame_num // window_size
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        listener_out = None
        for i in range(0, interval_num):
            if i != (interval_num-1):
                current_listener_in = input[:, i * window_size : (i + 1) * window_size]
            else:
                current_listener_in = input[:, i * window_size : ]

            current_listener_out = self.inner_model(current_listener_in * c_in, sigma, **kwargs) * c_out + input * c_skip
            if i != 0:
                listener_out = torch.cat((listener_out, current_listener_out), 1)
            else:
                listener_out = current_listener_out

        return listener_out

    def velocity_loss(self, inputs, outputs, window_size, mask = None):
        v_loss = self.sml1((outputs[:, 1:, :] - outputs[:, :-1, :]), (inputs[:, 1:, :] - inputs[:, :-1, :])) * mask.view(-1,
                                                                                                                     1,
                                                                                                                     1)
        i_loss = self.sml1((outputs[:, window_size:, :] - outputs[:, : -window_size, :]),
                           (inputs[:, window_size:, :] - inputs[:, : -window_size, :])) * mask.view(-1, 1, 1) / window_size

        v_loss = v_loss[:, :, 52:] * 10
        i_loss = i_loss[:, :, 52:] * 10

        if mask is not None:
            v_loss = v_loss * mask
            i_loss = i_loss * mask

        return v_loss.mean() + i_loss.mean()

    def kinematics_loss(self, inputs, target, mask = None):

        loss = 0

        for index in symmetry_actions:
            i, j = index
            loss += self.sml1(inputs[:, :, i] - inputs[:, :, j], target[:, :, i] - target[:, :, j])

        for index in co_occurred_actions:
            i, j = index
            loss += self.sml1(inputs[:, :, i] - inputs[:, :, j], target[:, :, i] - target[:, :, j])

        for index in exclusive_actions:
            i, j = index
            loss += self.sml1(inputs[:, :, i] - inputs[:, :, j], target[:, :, i] - target[:, :, j])

        if mask is not None:
            loss = loss * mask

        return loss.mean()


class DenoiserWithVariance(Denoiser):
    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
        logvar = utils.append_dims(logvar, model_output.ndim)
        target = (input - c_skip * noised_input) / c_out
        losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean(1)


class SimpleLossDenoiser(Denoiser):
    """L_simple with the Karras et al. preconditioner."""

    def loss(self, input, noise, sigma, **kwargs):
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        denoised = self(noised_input, sigma, **kwargs)
        eps = sampling.to_d(noised_input, sigma, denoised)
        return (eps - noise).pow(2).flatten(1).mean(1)


# Residual blocks

class ResidualBlock(nn.Module):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


# Noise level (and other) conditioning

class ConditionedModule(nn.Module):
    pass


class UnconditionedModule(ConditionedModule):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, cond=None):
        return self.module(input)


class ConditionedSequential(nn.Sequential, ConditionedModule):
    def forward(self, input, cond):
        for module in self:
            if isinstance(module, ConditionedModule):
                input = module(input, cond)
            else:
                input = module(input)
        return input


class DoubleIdentity(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return x, 0


class ConditionedResidualBlock(ConditionedModule):
    def __init__(self, *main, skip=None, rnn = None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else nn.Identity()
        self.rnn = rnn if rnn else DoubleIdentity()

    def forward(self, input, cond):
        skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
        x = self.main(input, cond)
        x, _ = self.rnn(x)
        return x + skip


class AdaGN(ConditionedModule):
    def __init__(self, feats_in, c_out, num_groups, eps=1e-5, cond_key='cond'):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_key = cond_key
        self.mapper = nn.Linear(feats_in, c_out * 2)

    def forward(self, input, cond):
        weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
        input = F.group_norm(input.transpose(1,2), self.num_groups, eps=self.eps)
        return torch.addcmul(bias.unsqueeze(-1), input, weight.unsqueeze(-1) + 1).transpose(1,2)



class TemporalAdaGN(ConditionedModule):
    def __init__(self, feats_in, c_out, num_groups, eps=1e-5, cond_key='cond_temporal'):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_key = cond_key
        self.mapper = nn.Linear(feats_in, c_out * 2)

    def forward(self, input, cond):
        weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
        input = F.group_norm(input.transpose(1,2), self.num_groups, eps=self.eps)
        return torch.addcmul(bias.transpose(1,2), input, weight.transpose(1,2) + 1).transpose(1,2)


# Attention

class SelfAttention1d(ConditionedModule):
    def __init__(self, c_in, n_head, norm, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm_in = norm(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Linear(c_in, c_in * 3)
        self.out_proj = nn.Linear(c_in, c_in)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, cond):
        n, t, c = input.shape
        qkv = self.qkv_proj(self.norm_in(input, cond))
        qkv = qkv.view([n, t, self.n_head * 3, c // self.n_head]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, t, c])
        return input + self.out_proj(y)


class CrossAttention1d(ConditionedModule):
    def __init__(self, c_dec, c_enc, n_head, norm_dec, dropout_rate=0.,
                 cond_key='speaker_3dmm', cond_key_padding='cross_padding'):
        super().__init__()
        assert c_dec % n_head == 0
        if cond_key == 'speaker_audio':
            self.pool = nn.MaxPool1d(2, 2)
        self.cond_key = cond_key
        self.cond_key_padding = cond_key_padding
        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = norm_dec(c_dec)
        self.n_head = n_head
        self.q_proj = nn.Linear(c_dec, c_dec)
        self.kv_proj = nn.Linear(c_enc, c_dec * 2)
        self.out_proj = nn.Linear(c_dec, c_dec)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, cond):
        n, t, c = input.shape
        if self.cond_key == 'speaker_audio':
            cross_cond = self.pool(cond[self.cond_key].transpose(1,2)).transpose(1,2).contiguous()
        else:
            cross_cond = cond[self.cond_key]
        q = self.q_proj(self.norm_dec(input, cond))
        q = q.view([n, t, self.n_head, c // self.n_head]).permute(0,2,1,3)
        kv = self.kv_proj(self.norm_enc(cross_cond))
        kv = kv.view([n,  t, self.n_head * 2, c // self.n_head]).permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, t, c])
        return input + self.out_proj(y)


# Downsampling/upsampling

_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic':
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3':
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
_kernels['bilinear'] = _kernels['linear']
_kernels['bicubic'] = _kernels['cubic']


class Downsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv2d(x, weight, stride=2)


class Upsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


# Embeddings

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class TemporalFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features, in_features * 2]) * std)

    def forward(self, input):
        f = input // 50
        f = torch.cat([f.cos(), f.sin()], dim=-1)
        f = f @ self.weight.T
        return f


# U-Nets

class UNet(ConditionedModule):
    def __init__(self, d_blocks, u_blocks, skip_stages=0):
        super().__init__()
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(u_blocks)
        self.skip_stages = skip_stages

    def forward(self, input, cond):
        skips = []
        for block in self.d_blocks[self.skip_stages:]:
            input = block(input, cond)
            skips.append(input)
        for i, (block, skip) in enumerate(zip(self.u_blocks, reversed(skips))):
            input = block(input, cond, skip if i > 0 else None)
        return input
