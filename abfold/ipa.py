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


from typing import Optional, Sequence

import math
import torch
import torch.nn as nn

from abfold.utils.rigid_utils import Rigid
from abfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


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
