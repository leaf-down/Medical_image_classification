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
                u1_cat_u2: torch.Tensor, u2_cat_u1: torch.Tensor,
                seqlen=None, seq_idx=None, cu_seqlens=None):

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
        BCdts1 = self.BCdts_in_proj(u1_cat_u2)
        BCdts2 = self.BCdts_in_proj(u2_cat_u1)
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
