# =============================================================================
# Name: CrossMamba_fusion
# Description: 基于使用Crossmamba融合方法的多模态图像融合
# Calling Attention:
"""
调用VFEFM模块时，cat_method = 'none'，不进行拼接，分支1直接使用分支2做参数输入，分支2一样
               cat_method = 'stack'，两分支在通道维度上进行堆叠，然后线性映射降维为原来的通道数。两分支使用相同的内容感知参数输入
               cat_method = 'add'，两分支直接相加做内容感知参数，两分支使用相同的内容感知参数输入
               cat_method = 'cls'，
"""
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as checkpoint

import math
from functools import partial
from typing import Callable

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# --------------------- CrossMamba Fusion Module ----------------------
class CrossMamba(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            d_model,  # 通道数
            d_state=128,
            # d_state="auto", # 20240109
            d_conv=3,  # 卷积核大小
            expand=2,
            headdim=64,
            d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
            ngroups=1,
            A_init_range=(1, 16),
            D_has_hdim=False,
            rmsnorm=True,
            norm_before_gate=False,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            dropout=0.,
            conv_bias=True,
            bias=False,
            chunk_size=256,
            use_mem_eff_path=True,
            layer_idx=None,  # Absorb kwarg for general module
            process_group=None,
            sequence_parallel=True,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group,
                                                sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
        # 跳跃连接相关量的输入映射(z0, x0, z)
        d_skip_in_proj = 2 * self.d_inner - self.d_ssm
        self.skip_in_proj = nn.Linear(self.d_model, d_skip_in_proj, bias=bias, **factory_kwargs)
        # 输入序列（xs）的输入映射
        self.xs_in_proj = nn.Linear(self.d_model, self.d_ssm, bias=bias, **factory_kwargs)
        # 内容感知相关参数的输入映射(Bs, Cs, dts)
        d_BCdts_in_proj = 2 * self.ngroups * self.d_state + self.nheads
        self.BCdts_in_proj = nn.Linear(self.d_model, d_BCdts_in_proj, bias=bias, **factory_kwargs)  # 输入维度未确定，后续更改

        conv_dim = (
                           self.d_ssm + 2 * self.ngroups * self.d_state) + self.nheads  # self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,  # 卷积核大小
            padding=(d_conv - 1) // 2,  # 填充
            **factory_kwargs,
        )
        # 输入序列的卷积
        self.xs_conv2d = nn.Conv2d(
            in_channels=self.d_ssm,
            out_channels=self.d_ssm,
            groups=self.d_ssm,
            bias=conv_bias,
            kernel_size=d_conv,  # 卷积核大小
            padding=(d_conv - 1) // 2,  # 填充
            **factory_kwargs,
        )
        # 内容感知相关参数的卷积
        d_BCdts_conv = 2 * self.ngroups * self.d_state + self.nheads
        self.BCdts_conv2d = nn.Conv2d(
            in_channels=d_BCdts_conv,
            out_channels=d_BCdts_conv,
            groups=d_BCdts_conv,
            bias=conv_bias,
            kernel_size=d_conv,  # 卷积核大小
            padding=(d_conv - 1) // 2,  # 填充
            **factory_kwargs,
        )
        self.act = nn.SiLU()  # 激活函数

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(torch.stack([inv_dt, inv_dt, inv_dt, inv_dt], dim=0))  # K=4
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # Initialize log A
        self.A_logs = self.A_log_init(A_init_range=A_init_range, nheads=self.nheads, dtype=dtype,
                                      copies=4)  # K=4

        # D "skip" parameter
        self.Ds = self.D_init(d_ssm=self.d_ssm, D_has_hdim=self.D_has_hdim, nheads=self.nheads,
                              copies=4)  # K=4

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group,
                                              sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def A_log_init(A_init_range, nheads, dtype, copies=1, device=None, merge=True):
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)  # Keep A_log in fp32  # 每个元素的对数
        if copies > 1:
            A_log = repeat(A_log, "n -> r n",
                           r=copies)  # 在新维度 r 上重复 copies 次，生成 (copies, d_inner, d_state) 的
            # A_log，为了在多方向扫描时为每个方向提供独立的参数初始化，使不同方向的 A_log 具有不同的初始化值。
            if merge:
                A_log = A_log.flatten(0,
                                      1)  # 将 A_log 的前两个维度 (copies, d_inner) 合并为一个维度，使得 A_log 的形状变为 (copies *
                # d_inner, d_state)，这种展平操作简化了后续计算，使得多方向的矩阵计算可以在同一个 A_log 中完成。
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_ssm, D_has_hdim, nheads, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_ssm if D_has_hdim else nheads, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, u1: torch.Tensor, u2: torch.Tensor,
                u2_cat_u1: torch.Tensor, u1_cat_u2: torch.Tensor,  # u2_cat_u1给u1用，u1_cat_u2给u2用，不要弄混
                seq_idx=None, cu_seqlens=None):

        B, H, W, C = u1.shape  # 批次数，通道数，高，宽
        L = H * W
        K = 4  # 四个方向上的扫描

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # 跳跃连接的输入映射
        zx1 = self.skip_in_proj(u1)
        zx2 = self.skip_in_proj(u2)
        # 分离x0和z0和z
        d_mlp = (self.d_inner - 2 * self.d_ssm) // 2
        z01, x01, z1 = torch.split(zx1,
                                   [d_mlp, d_mlp, self.d_ssm],
                                   dim=-1)
        z02, x02, z2 = torch.split(zx2,
                                   [d_mlp, d_mlp, self.d_ssm],
                                   dim=-1)

        # xs的输入映射
        xs1 = self.xs_in_proj(u1)
        xs2 = self.xs_in_proj(u2)
        xs1 = xs1.permute(0, 3, 1, 2).contiguous()  # 调整原张量顺序(b, c, h, w)
        xs2 = xs2.permute(0, 3, 1, 2).contiguous()
        xs1 = self.act(self.xs_conv2d(xs1))  # 卷积+激
        xs2 = self.act(self.xs_conv2d(xs2))

        # 内容感知相关参数的输入映射
        BCdts1 = self.BCdts_in_proj(u2_cat_u1)
        BCdts2 = self.BCdts_in_proj(u1_cat_u2)
        BCdts1 = BCdts1.permute(0, 3, 1, 2).contiguous()  # 调整原张量顺序(b, c, h, w)
        BCdts2 = BCdts2.permute(0, 3, 1, 2).contiguous()
        BCdts1 = self.act(self.BCdts_conv2d(BCdts1))  # 卷积+激活
        BCdts2 = self.act(self.BCdts_conv2d(BCdts2))

        # 对输入和内容感知参数进行4方向扫描
        # 先通道上拼接然后统一进行再通道上拆分
        xBCdts1 = torch.cat([xs1, BCdts1], dim=1)
        xBCdts2 = torch.cat([xs2, BCdts2], dim=1)
        # x 展平为大小 (B, C, L)，特征图的高度和宽度转置同样展平，两种排列堆叠起来，生成一个 (B, 2, C, L) 的张量。
        xBCdt_hwwh1 = torch.stack(
            [xBCdts1.view(B, -1, L), torch.transpose(xBCdts1, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)  # -1这个参数是让pytorch自动推断维度的大小，确保总元素数不变
        xBCdt1 = torch.cat([xBCdt_hwwh1, torch.flip(xBCdt_hwwh1, dims=[-1])],
                           dim=1)  # (b, k, c, l)  生成正向和逆向的特征排列（即翻转最后一维）最终四种组合
        xBCdt_hwwh2 = torch.stack(
            [xBCdts2.view(B, -1, L), torch.transpose(xBCdts2, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)  # -1这个参数是让pytorch自动推断维度的大小，确保总元素数不变
        xBCdt2 = torch.cat([xBCdt_hwwh2, torch.flip(xBCdt_hwwh2, dims=[-1])],
                           dim=1)  # (b, k, c, l)  生成正向和逆向的特征排列（即翻转最后一维）最终四种组合
        # 第二次分离
        xBC1, dt1 = torch.split(xBCdt1, [self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads], dim=2)
        x1, B1, C1 = torch.split(xBC1, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                 dim=2)
        xBC2, dt2 = torch.split(xBCdt2, [self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads], dim=2)
        x2, B2, C2 = torch.split(xBC2, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                 dim=2)

        # (b, k * c, l)，平展，匹配mamba的序列维度
        # (b, l, k * c), 匹配mamba2的形状要求
        x1 = x1.float().reshape(B, -1, L).permute(0, 2, 1)
        B1 = B1.float().reshape(B, -1, L).permute(0, 2, 1)
        C1 = C1.float().reshape(B, -1, L).permute(0, 2, 1)
        dt1 = dt1.float().reshape(B, -1, L).permute(0, 2, 1)

        x2 = x2.float().reshape(B, -1, L).permute(0, 2, 1)
        B2 = B2.float().reshape(B, -1, L).permute(0, 2, 1)
        C2 = C2.float().reshape(B, -1, L).permute(0, 2, 1)
        dt2 = dt2.float().reshape(B, -1, L).permute(0, 2, 1)

        dt_bias = self.dt_bias.view(-1)

        # xs
        x1 = rearrange(x1, "b l (h p) -> b l h p", p=self.headdim)
        x2 = rearrange(x2, "b l (h p) -> b l h p", p=self.headdim)
        # As
        As = -torch.exp(self.A_logs.float())
        # Bs
        B1 = rearrange(B1, "b l (g n) -> b l g n", g=self.ngroups)
        B2 = rearrange(B2, "b l (g n) -> b l g n", g=self.ngroups)
        # Cs
        C1 = rearrange(C1, "b l (g n) -> b l g n", g=self.ngroups)
        C2 = rearrange(C2, "b l (g n) -> b l g n", g=self.ngroups)
        # Ds
        Ds = rearrange(self.Ds, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.Ds

        def mamba_core(xs, dts, Bs, Cs, z, z0, x0):
            y = mamba_chunk_scan_combined(
                xs,
                dts,
                As,
                Bs,
                Cs,
                chunk_size=self.chunk_size,
                D=Ds,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
            )  # (b, l, k, d)与xBCdts形状一致
            y = rearrange(y, "b l h p -> b l (h p)")
            y = y.view(B, L, K, -1)  # (b, l, k, d)与xBCdts形状一致
            assert y.dtype == torch.float

            out_y = y[:, :, 0]  # 第一个方向
            inv_y = torch.flip(y[:, :, 2:4], dims=[1]).view(B, L, 2, -1)  # 先第三和第四方向
            wh_y = torch.transpose(y[:, :, 1].view(B, W, H, -1), dim0=1, dim1=2).contiguous().view(B, L, -1)  # 第二个方向
            invwh_y = torch.transpose(inv_y[:, :, 1].view(B, W, H, -1), dim0=1, dim1=2).contiguous().view(B, L,
                                                                                                          -1)  # 第四个方向

            y1 = out_y
            y2 = inv_y[:, :, 0]
            y3 = wh_y
            y4 = invwh_y

            out = y1 + y2 + y3 + y4
            out = out.contiguous().view(B, H, W, -1)

            if self.rmsnorm:
                out = self.norm(out, z)
            if d_mlp > 0:
                out = torch.cat([F.silu(z0) * x0, out], dim=-1)

            out_data = self.out_proj(out)
            if self.dropout is not None:
                out_data = self.dropout(out_data)

            return out_data

        out1 = mamba_core(xs=x1,
                          dts=dt1,
                          Bs=B1,
                          Cs=C1,
                          z=z1,
                          z0=z01,
                          x0=x01)
        out2 = mamba_core(xs=x2,
                          dts=dt2,
                          Bs=B2,
                          Cs=C2,
                          z=z2,
                          z0=z02,
                          x0=x02)
        return out1, out2


# --------------------- MedSSD Feature Extraction Module ----------------------

class MedSSD(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            d_model,  # 通道数
            d_state=128,
            # d_state="auto", # 20240109
            d_conv=3,  # 卷积核大小
            expand=2,
            headdim=64,
            d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
            ngroups=1,
            A_init_range=(1, 16),
            D_has_hdim=False,
            rmsnorm=True,
            norm_before_gate=False,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            dropout=0.,
            conv_bias=True,
            bias=False,
            chunk_size=256,
            use_mem_eff_path=True,
            layer_idx=None,  # Absorb kwarg for general module
            process_group=None,
            sequence_parallel=True,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group,
                                                sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = (
                           self.d_ssm + 2 * self.ngroups * self.d_state) + self.nheads  # self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,  # 卷积核大小
            padding=(d_conv - 1) // 2,  # 填充
            **factory_kwargs,
        )
        self.act = nn.SiLU()  # 激活函数

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(torch.stack([inv_dt, inv_dt, inv_dt, inv_dt], dim=0))  # K=4
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # Initialize log A
        self.A_logs = self.A_log_init(A_init_range=A_init_range, nheads=self.nheads, dtype=dtype,
                                      copies=4)  # K=4

        # D "skip" parameter
        self.Ds = self.D_init(d_ssm=self.d_ssm, D_has_hdim=self.D_has_hdim, nheads=self.nheads,
                              copies=4)  # K=4

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group,
                                              sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def A_log_init(A_init_range, nheads, dtype, copies=1, device=None, merge=True):
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)  # Keep A_log in fp32  # 每个元素的对数
        if copies > 1:
            A_log = repeat(A_log, "n -> r n",
                           r=copies)  # 在新维度 r 上重复 copies 次，生成 (copies, d_inner, d_state) 的
            # A_log，为了在多方向扫描时为每个方向提供独立的参数初始化，使不同方向的 A_log 具有不同的初始化值。
            if merge:
                A_log = A_log.flatten(0,
                                      1)  # 将 A_log 的前两个维度 (copies, d_inner) 合并为一个维度，使得 A_log 的形状变为 (copies *
                # d_inner, d_state)，这种展平操作简化了后续计算，使得多方向的矩阵计算可以在同一个 A_log 中完成。
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_ssm, D_has_hdim, nheads, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_ssm if D_has_hdim else nheads, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, u: torch.Tensor, seqlen=None, seq_idx=None, cu_seqlens=None):

        B, H, W, C = u.shape  # 批次数，通道数，高，宽
        L = H * W
        K = 4  # 四个方向上的扫描

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        zxbcdt = self.in_proj(u)
        # 相比于mamba1，mamba2将所有的变量一起进行了投影
        # 对于K=4的操作来说，x0和z0和z用于跳跃连接，因而不操作
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        # 分离时先将x0和z0和z单独分离出去，剩下的K=4操作后再分离，用括号标注了后面两项的分离
        z0, x0, z, xBCdt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, (self.d_ssm + 2 * self.ngroups * self.d_state) + self.nheads],
            dim=-1
        )

        xBCdt = xBCdt.permute(0, 3, 1, 2).contiguous()  # 调整原张量顺序(b, c, h, w)
        xBCdt = self.act(self.conv2d(xBCdt))

        # x 展平为大小 (B, C, L)，特征图的高度和宽度转置同样展平，两种排列堆叠起来，生成一个 (B, 2, C, L) 的张量。
        xBCdt_hwwh = torch.stack(
            [xBCdt.view(B, -1, L), torch.transpose(xBCdt, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)  # -1这个参数是让pytorch自动推断维度的大小，确保总元素数不变
        xBCdts = torch.cat([xBCdt_hwwh, torch.flip(xBCdt_hwwh, dims=[-1])],
                           dim=1)  # (b, k, d, l)  生成正向和逆向的特征排列（即翻转最后一维）最终四种组合

        # 第二次分离
        xBCs, dts = torch.split(xBCdts, [self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads], dim=2)
        xs, Bs, Cs = torch.split(xBCs, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=2)

        # (b, k * d, l)，平展，匹配mamba的序列维度
        # (b, l, k * d), 匹配mamba2的形状要求
        xs = xs.float().reshape(B, -1, L).permute(0, 2, 1)
        Bs = Bs.float().reshape(B, -1, L).permute(0, 2, 1)
        Cs = Cs.float().reshape(B, -1, L).permute(0, 2, 1)
        dts = dts.float().reshape(B, -1, L).permute(0, 2, 1)
        dt_bias = self.dt_bias.view(-1)

        # xs
        xs = rearrange(xs, "b l (h p) -> b l h p", p=self.headdim)
        # As
        As = -torch.exp(self.A_logs.float())
        # Bs
        Bs = rearrange(Bs, "b l (g n) -> b l g n", g=self.ngroups)
        # Cs
        Cs = rearrange(Cs, "b l (g n) -> b l g n", g=self.ngroups)
        # Ds
        Ds = rearrange(self.Ds, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.Ds

        y = mamba_chunk_scan_combined(
            xs,
            dts,
            As,
            Bs,
            Cs,
            chunk_size=self.chunk_size,
            D=Ds,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
        )  # (b, l, k, d)与xBCdts形状一致
        y = rearrange(y, "b l h p -> b l (h p)")
        y = y.view(B, L, K, -1)  # (b, l, k, d)与xBCdts形状一致
        assert y.dtype == torch.float

        out_y = y[:, :, 0]  # 第一个方向
        inv_y = torch.flip(y[:, :, 2:4], dims=[1]).view(B, L, 2, -1)  # 先第三和第四方向
        wh_y = torch.transpose(y[:, :, 1].view(B, W, H, -1), dim0=1, dim1=2).contiguous().view(B, L, -1)  # 第二个方向
        invwh_y = torch.transpose(inv_y[:, :, 1].view(B, W, H, -1), dim0=1, dim1=2).contiguous().view(B, L, -1)  # 第四个方向

        y1 = out_y
        y2 = inv_y[:, :, 0]
        y3 = wh_y
        y4 = invwh_y

        out = y1 + y2 + y3 + y4
        out = out.contiguous().view(B, H, W, -1)

        if self.rmsnorm:
            out = self.norm(out, z)
        if d_mlp > 0:
            out = torch.cat([F.silu(z0) * x0, out], dim=-1)

        out_data = self.out_proj(out)
        if self.dropout is not None:
            out_data = self.dropout(out_data)

        return out_data


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x


class SS_Conv_SSD(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim // 2)
        self.self_attention = MedSSD(d_model=hidden_dim // 2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2, dim=-1)
        x = self.drop_path(self.self_attention(self.ln_1(input_right)))
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        input_left = self.conv33conv33conv11(input_left)
        input_left = input_left.permute(0, 2, 3, 1).contiguous()
        output = torch.cat((input_left, x), dim=-1)
        output = channel_shuffle(output, groups=2)
        return output + input


# --------------------- Embedding, Downsampling, Upsampling Module ----------------------
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


# --------------------- Overall Structure ----------------------
class downLayer(nn.Module, PyTorchModelHubMixin):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            cat_method,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=64,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.cat_method = cat_method
        self.cat_proj = None
        if self.cat_method == 'stack':
            self.cat_proj = nn.Linear(dim * 2, dim)
        elif self.cat_method == 'cls':
            self.cat_proj = nn.Linear(dim, dim)  # 该方法未确定，确定后更改

        self.blocks1 = nn.ModuleList([
            SS_Conv_SSD(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        self.blocks2 = nn.ModuleList([
            SS_Conv_SSD(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        self.fusion = CrossMamba(d_model=dim, dropout=attn_drop)

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample1 = downsample(dim=dim, norm_layer=norm_layer)
            self.downsample2 = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample1 = None
            self.downsample2 = None

    def forward(self, x1, x2):
        for blk in self.blocks1:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1)
            else:
                x1 = blk(x1)

        for blk in self.blocks2:
            if self.use_checkpoint:
                x2 = checkpoint.checkpoint(blk, x2)
            else:
                x2 = blk(x2)

        x2_cat_x1 = x2
        x1_cat_x2 = x1
        if self.cat_method == 'none':
            # 不进行拼接操作，直接使用原序列
            x2_cat_x1 = x2
            x1_cat_x2 = x1
        elif self.cat_method == 'add':
            x2_cat_x1 = x1 + x2
            x1_cat_x2 = x1 + x2
        elif self.cat_method == 'stack':
            u = torch.cat([x1, x2], dim=-1)
            u = self.cat_proj(u)
            x2_cat_x1 = u
            x1_cat_x2 = u
        elif self.cat_method == 'cls':
            pass

        x1_f, x2_f = self.fusion(x1, x2, x2_cat_x1, x1_cat_x2)
        x1_f = x1 + x1_f
        x2_f = x2 + x2_f

        if self.downsample is not None:
            x1_f = self.downsample1(x1_f)
            x2_f = self.downsample2(x2_f)

        return x1_f, x2_f


class upLayer(nn.Module, PyTorchModelHubMixin):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            cat_method,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            skip=True,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.cat_method = cat_method
        self.cat_proj = None
        if self.cat_method == 'stack':
            self.cat_proj = nn.Linear(dim * 2, dim)
        elif self.cat_method == 'cls':
            self.cat_proj = nn.Linear(dim, dim)  # 该方法未确定，确定后更改

        self.in_proj1 = nn.Linear(dim * 2, dim)
        self.in_proj2 = nn.Linear(dim * 2, dim)

        self.blocks1 = nn.ModuleList([
            SS_Conv_SSD(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        self.blocks2 = nn.ModuleList([
            SS_Conv_SSD(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        self.fusion = CrossMamba(d_model=dim, dropout=attn_drop)

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample1 = upsample(dim=dim, norm_layer=norm_layer)
            self.upsample2 = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample1 = None
            self.upsample2 = None

        self.skip = skip

    def forward(self, x10, x20,
                x1_down, x2_down):

        if self.skip:
            x10 = torch.cat([x10, x1_down], dim=-1)
            x1 = self.in_proj1(x10)
            x20 = torch.cat([x20, x2_down], dim=-1)
            x2 = self.in_proj2(x20)
        else:
            x1 = x10
            x2 = x20

        for blk in self.blocks1:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1)
            else:
                x1 = blk(x1)

        for blk in self.blocks2:
            if self.use_checkpoint:
                x2 = checkpoint.checkpoint(blk, x2)
            else:
                x2 = blk(x2)

        x2_cat_x1 = x2
        x1_cat_x2 = x1
        if self.cat_method == 'none':
            # 不进行拼接操作，直接使用原序列
            x2_cat_x1 = x2
            x1_cat_x2 = x1
        elif self.cat_method == 'add':
            x2_cat_x1 = x1 + x2
            x1_cat_x2 = x1 + x2
        elif self.cat_method == 'stack':
            u = torch.cat([x1, x2], dim=-1)
            u = self.cat_proj(u)
            x2_cat_x1 = u
            x1_cat_x2 = u
        elif self.cat_method == 'cls':
            pass

        x1_f, x2_f = self.fusion(x1, x2, x2_cat_x1, x1_cat_x2)
        x1_f = x1 + x1_f
        x2_f = x2 + x2_f

        if self.downsample is not None:
            x1_f = self.upsample1(x1_f)
            x2_f = self.upsample2(x2_f)

        return x1_f, x2_f


class VFEFM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 depths=[2, 2, 4, 2],
                 dims=[128, 256, 512, 1024],  # 原为[96, 192, 384, 768]
                 depths_decoder=[2, 9, 2, 2],
                 dims_decoder=[1024, 512, 256, 128],
                 d_state=128, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 cat_method='none',  # 非常重要，调用时记得填写
                 **kwargs):
        super().__init__()

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_drop2 = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.norm = nn.LayerNorm(dims_decoder[-1] * 2)

        # ------------------------------------ 下采样部分 ------------------------------------------
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed1 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                         norm_layer=norm_layer if patch_norm else None)
        self.patch_embed2 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                         norm_layer=norm_layer if patch_norm else None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = downLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                cat_method=cat_method,
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,  # 只有最后一层不含下采样
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # ------------------------------------- 上采样部分 -------------------------------------------
        self.num_layers_up = len(depths_decoder)
        self.final_dim = dims_decoder[-1]
        self.dims_decoder = dims_decoder
        self.final_cat_proj = nn.Linear(self.final_dim * 2, self.final_dim)

        self.final_expand = Final_PatchExpand2D(dim=dims_decoder[-1])
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, 1, 1)

        self.layers_up = nn.ModuleList()
        for i_layer_up in range(self.num_layers_up):
            layer_up = upLayer(
                dim=dims_decoder[i_layer_up],
                depth=depths_decoder[i_layer_up],
                cat_method=cat_method,
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_decoder[:i_layer_up]):sum(depths_decoder[:i_layer_up + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer_up < self.num_layers_up - 1) else None,  # 只有最后一层不含下采样
                skip=False if (i_layer_up == 0) else True,  # 上采样第一层因直接连接下采样最后一层故不使用跳跃连接
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer_up)

        # ------------------------------------ 桥接部分 -------------------------------------
        self.bridge1 = nn.Conv2d(dims[-1], dims_decoder[0], 1)  # 目前仅作占位，后续加新的设计
        self.bridge2 = nn.Conv2d(dims[-1], dims_decoder[0], 1)

        # -------------------------------- 模型常规参数初始化 -------------------------------------------
        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in SS_Conv_SSD, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, SS_Conv_SSD initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_down(self, x1, x2):
        x1 = self.patch_embed1(x1)
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
        x1 = self.pos_drop1(x1)

        x2 = self.patch_embed2(x2)
        if self.ape:
            x2 = x2 + self.absolute_pos_embed
        x2 = self.pos_drop2(x2)

        skip = []
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
            skip.append((x1, x2))

        return x1, x2, skip

    def forward_up(self, x1, x2, skip):
        # 桥接部分等待一个新的设计
        x1 = self.bridge1(x1)
        x2 = self.bridge2(x2)

        i = 0
        for layer_up in self.layers_up:
            u1 = skip[i][0]
            u2 = skip[i][1]
            x1, x2 = layer_up(x1, x2, u1, u2)

        x = self.norm(torch.cat([x1, x2], dim=-1))
        x_cat = self.final_cat_proj(x)  # 一个用来降维加两个通道特征融合的模块（计划用KAN，目前先写linear）

        # 最后连一个Final_Expand
        out = self.final_expand(x_cat)

        return out

    def forward(self, x1, x2):
        x1, x2, skip = self.forward_down(x1, x2)
        x = self.forward_up(x1, x2, skip)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)

        return x
