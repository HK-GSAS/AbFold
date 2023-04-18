# Copyright 2022 Ahdritz, Gustaf and Bouatta, Nazim and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import math
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from abfold.utils.rigid_utils import Rigid
from typing import Optional, Sequence
from abfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
from invariant_point_attention.invariant_point_attention import exists, disable_tf32, default, max_neg_value, \
    FeedForward, InvariantPointAttention as SelfInvariantPointAttention, IPABlock as SelfIPABlock


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
            self,
            c_s: int,
            c_z: int,
            c_hidden: int,
            no_heads: int,
            no_qk_points: int,
            no_v_points: int,
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = nn.Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
                self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = nn.Linear(concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s_q: torch.Tensor,
            s_kv: torch.Tensor,
            z: Optional[torch.Tensor],
            r_q: Rigid,
            r_kv: Rigid,
            mask_q: torch.Tensor,
            mask_kv: torch.Tensor,
            inplace_safe: bool = False,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s_q:
                [*, N_res_q, C_s] single representation
            s_kv:
                [*, N_res_kv, C_s] single representation
            z:
                [*, N_res_q, N_res_kv, C_z] pair representation
            r_q:
                [*, N_res_q] transformation object
            r_kv:
                [*, N_res_kv] transformation object
            mask_q:
                [*, N_res_q] mask
            mask_kv:
                [*, N_res_kv] mask
        Returns:
            [*, N_res_q, C_s] single representation update
        """
        if (_offload_inference and inplace_safe):
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res_q, H * C_hidden]
        q = self.linear_q(s_q)
        # [*, N_res_kv, H * C_hidden]
        kv = self.linear_kv(s_kv)

        # [*, N_res_q, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res_kv, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res_kv, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res_q, H * P_q * 3]
        q_pts = self.linear_q_points(s_q)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res_q, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r_q[..., None].apply(q_pts)

        # [*, N_res_q, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res_kv, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s_kv)

        # [*, N_res_kv, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r_kv[..., None].apply(kv_pts)

        # [*, N_res_kv, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res_kv, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res_q, N_res_kv, H]
        b = self.linear_b(z[0])

        if (_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res_q, N_res_kv]
        a = torch.matmul(
            permute_final_dims(q, [1, 0, 2]),  # [*, H, N_res_q, C_hidden]
            permute_final_dims(k, [1, 2, 0]),  # [*, H, C_hidden, N_res_kv]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, [2, 0, 1]))

        # [*, N_res_q, N_res_kv, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if (inplace_safe):
            pt_att *= pt_att
        else:
            pt_att = pt_att ** 2

        # [*, N_res_q, N_res_kv, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        if (inplace_safe):
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [*, N_res_q, N_res_kv, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res_q, N_res_kv]
        square_mask = mask_q.unsqueeze(-1) * mask_kv.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res_q, N_res_kv]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res_q, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res_q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res_q, P_v]
        if (inplace_safe):
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [
                torch.matmul(a, v.to(a.dtype))
                for v in torch.unbind(v_pts, dim=-3)
            ]
            o_pt = torch.stack(o_pt, dim=-3)
        else:
            o_pt = torch.sum(
                (
                        a[..., None, :, :, None]
                        * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )

        # [*, N_res_q, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r_q[..., None, None].invert_apply(o_pt)

        # [*, N_res_q, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res_q, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res_q, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res_q, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res_q, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class CrossInvariantPointAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            scalar_key_dim=16,
            scalar_value_dim=16,
            point_key_dim=4,
            point_value_dim=4,
            pairwise_repr_dim=None,
            require_pairwise_repr=True,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias=False)

        # pairwise representation projection to attention bias

        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads),
                Rearrange('b ... h -> (b h) ...')
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)

        self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(
            self,
            q_single_repr,
            kv_single_repr,
            pairwise_repr=None,
            *,
            rotations,
            translations,
            q_mask=None,
            kv_mask=None
    ):
        q_x, b, h, eps, require_pairwise_repr = q_single_repr, q_single_repr.shape[
            0], self.heads, self.eps, self.require_pairwise_repr
        kv_x = kv_single_repr
        assert not (require_pairwise_repr and not exists(
            pairwise_repr)), 'pairwise representation must be given as second argument'

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways

        q_scalar, k_scalar, v_scalar = self.to_scalar_q(q_x), self.to_scalar_k(kv_x), self.to_scalar_v(kv_x)

        q_point, k_point, v_point = self.to_point_q(q_x), self.to_point_k(kv_x), self.to_point_v(kv_x)

        # split out heads

        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                                           (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h=h, c=3),
                                        (q_point, k_point, v_point))

        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h=h)
        translations = repeat(translations, 'b n c -> (b h) n () c', h=h)

        # rotate qkv points into global frame

        q_point = einsum('b n d c, b n c r -> b n d r', q_point, rotations) + translations
        k_point = einsum('b n d c, b n c r -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n c r -> b n d r', v_point, rotations) + translations

        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale

        if require_pairwise_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim=(-1, -2))

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () ()', b=b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)

        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        # mask

        if exists(q_mask) and exists(kv_mask):
            mask = rearrange(q_mask, 'b i -> b i ()') * rearrange(kv_mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention

        attn = attn_logits.softmax(dim=- 1)

        with disable_tf32(), autocast(enabled=False):
            # disable TF32 for precision

            # aggregate values

            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h=h)

            if require_pairwise_repr:
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values

            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)

            # rotate aggregated point values back into local frame

            results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations,
                                    rotations.transpose(-1, -2))
            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h=h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h=h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h=h)

        results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h=h)
            results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim=-1)
        return self.to_out(results)


class CrossIPABlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            ff_mult=1,
            ff_num_layers=3,  # in the paper, they used 3 layer transition (feedforward) block
            post_norm=True,  # in the paper, they used post-layernorm - offering pre-norm as well
            post_attn_dropout=0.,
            post_ff_dropout=0.,
            **kwargs
    ):
        super().__init__()
        self.post_norm = post_norm
        self.q_attn_norm = nn.LayerNorm(dim)
        self.kv_attn_norm = nn.LayerNorm(dim)
        self.attn = CrossInvariantPointAttention(dim=dim, **kwargs)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, num_layers=ff_num_layers)
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)

    def forward(self, q_x, kv_x, **kwargs):
        post_norm = self.post_norm
        if post_norm:
            q_attn_input = q_x
            kv_attn_input = kv_x
        else:
            q_attn_input = self.q_attn_norm(q_x)
            kv_attn_input = self.kv_attn_norm(kv_x)
        x = self.attn(q_attn_input, kv_attn_input, **kwargs) + q_x
        x = self.post_attn_dropout(x)
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.post_ff_dropout(x)
        if post_norm:
            x = self.ff_norm(x)
        return x


class IPATransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            num_tokens=None,
            predict_points=False,
            detach_rotations=True,
            **kwargs
    ):
        super().__init__()

        # using quaternion functions from pytorch3d

        self.quaternion_to_matrix = quaternion_to_matrix
        self.quaternion_multiply = quaternion_multiply

        # layers

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                IPABlock(dim=dim, **kwargs),
                nn.Linear(dim, 6)
            ]))

        # whether to detach rotations or not, for stability during training

        self.detach_rotations = detach_rotations

        # output

        self.predict_points = predict_points

        if predict_points:
            self.to_points = nn.Linear(dim, 3)

    def forward(
            self,
            single_repr,
            *,
            translations=None,
            quaternions=None,
            pairwise_repr=None,
            mask=None
    ):
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        if exists(self.token_emb):
            x = self.token_emb(x)

        # if no initial quaternions passed in, start from identity

        if not exists(quaternions):
            quaternions = torch.tensor([1., 0., 0., 0.], device=device)  # initial rotations
            quaternions = repeat(quaternions, 'd -> b n d', b=b, n=n)

        # if not translations passed in, start from identity

        if not exists(translations):
            translations = torch.zeros((b, n, 3), device=device)

        # go through the layers and apply invariant point attention and feedforward

        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)

            if self.detach_rotations:
                rotations.detach_()

            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations
            )

            # update quaternion and translation

            quaternion_update, translation_update = to_update(x).chunk(2, dim=-1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)

            quaternions = quaternion_multiply(quaternions, quaternion_update)
            translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

        if not self.predict_points:
            return x, translations, quaternions

        points_local = self.to_points(x)
        rotations = quaternion_to_matrix(quaternions)
        points_global = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations
        return points_global
