from typing import Optional, List, Tuple

import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Attention as SelfAttention
from timm.models.vision_transformer import LayerScale


# 调换最后 inds 维度 例：tensor [1,2,3,4,5] inds [1,0] => tensor [1,2,3,5,4]
def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


# @torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, [1, 0])

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = torch.nn.functional.softmax(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
            self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,
            gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = nn.Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = nn.Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = nn.Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = nn.Linear(
            self.c_hidden * self.no_heads, self.c_q
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
                 o: torch.Tensor,
                 q_x: torch.Tensor
                 ) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, H, Q, C_hidden]
        o = _attention(q, k, v, biases)

        # [*, Q, H, C_hidden]
        o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_x, kv_x):
        B, q_N, C = q_x.shape
        kv_N = kv_x.shape[1]
        q = self.q(q_x).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv_x).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q_x, kv_x):
        kv_x = self.norm_kv(kv_x)
        q_x = q_x + self.drop_path1(self.ls1(self.attn(self.norm1(q_x), kv_x)))
        q_x = q_x + self.drop_path2(self.ls2(self.mlp(self.norm2(q_x))))
        return q_x


class AttentionModule(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.h_self = SelfAttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.l_self = SelfAttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.hl_cross = CrossAttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.lh_cross = CrossAttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

    def forward(self, h_feat, l_feat):
        h_feat = self.h_self(h_feat)
        l_feat = self.l_self(l_feat)
        l_feat = self.lh_cross(l_feat, h_feat)
        h_feat = self.hl_cross(h_feat, l_feat)

        return h_feat, l_feat


class BiasAttention(nn.Module):
    def __init__(self, c_q, c_kv, head_dim, num_heads, bias_dim):
        super(BiasAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(c_kv, 2 * num_heads * head_dim)
        self.q = nn.Linear(c_q, num_heads * head_dim)
        self.bias = nn.Linear(bias_dim, num_heads)
        self.proj = nn.Linear(num_heads * head_dim, c_q)

    def forward(self, x_q, x_kv, z=None):
        B, N, C_kv = x_kv.shape
        N_q = x_q.shape[1]
        kv = self.kv(x_kv).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # [B, H, N, head_dim]
        k, v = kv.unbind(0)
        q = self.q(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if z is not None:
            b = self.bias(z).permute(0, 3, 1, 2)
            attn = attn + b
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, self.num_heads * self.head_dim)
        x = self.proj(x)

        return x


class BiasAttentionBlock(nn.Module):
    def __init__(self, c_q, c_kv, head_dim, num_heads, bias_dim):
        super(BiasAttentionBlock, self).__init__()

        self.norm1 = nn.LayerNorm(c_q)
        self.attn = BiasAttention(c_q, c_kv, head_dim, num_heads, bias_dim)

        self.norm2 = nn.LayerNorm(c_q)
        self.mlp = Mlp(in_features=c_q, hidden_features=4 * c_q)

    def forward(self, x_q, x_kv, z=None):
        x_q = x_q + self.attn(self.norm1(x_q), x_kv, z)
        x_q = x_q + self.mlp(self.norm2(x_q))

        return x_q


class BiasAttentionModule(nn.Module):
    def __init__(self, c_q, c_kv, c_point, head_dim, num_heads, bias_dim):
        super(BiasAttentionModule, self).__init__()

        self.h_self = BiasAttentionBlock(c_q, c_kv, head_dim, num_heads, bias_dim)
        self.l_self = BiasAttentionBlock(c_q, c_kv, head_dim, num_heads, bias_dim)
        self.h_point_cross = BiasAttentionBlock(c_q, c_point, head_dim, num_heads, bias_dim)
        self.l_point_cross = BiasAttentionBlock(c_q, c_point, head_dim, num_heads, bias_dim)
        self.hl_cross = BiasAttentionBlock(c_q, c_kv, head_dim, num_heads, bias_dim)
        self.lh_cross = BiasAttentionBlock(c_q, c_kv, head_dim, num_heads, bias_dim)

    def forward(self, h_feat, l_feat, z_hh, z_hl, z_ll, z_lh, point_feat):
        h_feat = self.h_self(h_feat, h_feat, z_hh)
        l_feat = self.l_self(l_feat, l_feat, z_ll)
        h_feat = self.h_point_cross(h_feat, point_feat)
        l_feat = self.l_point_cross(l_feat, point_feat)
        l_feat = self.lh_cross(l_feat, h_feat, z_lh)
        h_feat = self.hl_cross(h_feat, l_feat, z_hl)

        return h_feat, l_feat
