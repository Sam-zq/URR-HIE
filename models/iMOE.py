import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Expert(nn.Module):
    """
    单个专家网络：标准的双层前馈神经网络 (FFN)
    """

    def __init__(self, d_model, d_ff):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class PhysicsGuidedMoE(nn.Module):
    """
    基于 GHI Z-score 物理硬约束引导的混合专家层
    """

    def __init__(self, configs, num_experts=3):
        super(PhysicsGuidedMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(configs.d_model, configs.d_ff) for _ in range(num_experts)])

        # 门控网络：输入为单维度的 GHI 统计值
        self.gate = nn.Linear(1, num_experts)

        # 从 configs 接收 GHI 的真实均值和标准差 (带防崩溃默认值)
        self.ghi_mu = getattr(configs, 'ghi_mu', 0.0)
        self.ghi_sigma = getattr(configs, 'ghi_sigma', 1.0)

        # 预设物理真实阈值 (W/m^2)
        self.phys_threshold_low = 150.0  # 区分弱光和中光
        self.phys_threshold_high = 600.0  # 区分中光和强光

        # 反算对应的 Z-score 阈值
        self.z_threshold_low = (self.phys_threshold_low - self.ghi_mu) / self.ghi_sigma
        self.z_threshold_high = (self.phys_threshold_high - self.ghi_mu) / self.ghi_sigma

    def forward(self, x, ghi_mean_z):
        B, C, D = x.shape

        # 1. 计算初始门控得分
        gate_logits = self.gate(ghi_mean_z.unsqueeze(-1))  # [B, 3]

        # 2. 注入 Z-score 物理硬约束掩码
        mask = torch.zeros_like(gate_logits)
        mask[ghi_mean_z < self.z_threshold_low, 2] = -1e9  # 极暗环境屏蔽强光专家
        mask[ghi_mean_z > self.z_threshold_high, 0] = -1e9  # 极亮环境屏蔽弱光专家

        gate_logits = gate_logits + mask
        routing_weights = F.softmax(gate_logits, dim=-1)

        # 3. Top-1 稀疏路由分配
        top1_weights, top1_indices = torch.max(routing_weights, dim=-1)
        final_output = torch.zeros_like(x)

        # 4. 专家并行计算 (仅计算被激活的部分)
        for i, expert in enumerate(self.experts):
            batch_idx = (top1_indices == i).nonzero(as_tuple=True)[0]
            if batch_idx.numel() > 0:
                expert_input = x[batch_idx]
                expert_output = expert(expert_input) * top1_weights[batch_idx].view(-1, 1, 1)
                final_output[batch_idx] = expert_output

        return final_output


class MoE_EncoderLayer(nn.Module):
    """
    魔改后的编码器层：将标准 FFN 替换为 PhysicsGuidedMoE
    """

    def __init__(self, configs):
        super(MoE_EncoderLayer, self).__init__()

        # 调用 tslib 原生的注意力机制层 (用于捕获多变量之间的依赖)
        self.attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=False), configs.d_model, configs.n_heads)

        # 核心替换：物理约束 MoE
        self.moe = PhysicsGuidedMoE(configs, num_experts=3)

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, ghi_mean_z):
        # 注意力机制 (变量维度)
        new_x, _ = self.attention(x, x, x, attn_mask=None)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 深层物理门控 MoE
        moe_out = self.moe(x, ghi_mean_z)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)

        return x


class MoE_Encoder(nn.Module):
    """
    魔改后的编码器堆叠容器：负责将 ghi_mean_z 传递给每一层
    """

    def __init__(self, configs):
        super(MoE_Encoder, self).__init__()
        self.layers = nn.ModuleList([MoE_EncoderLayer(configs) for _ in range(configs.e_layers)])
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x, ghi_mean_z):
        for layer in self.layers:
            x = layer(x, ghi_mean_z)  # 核心：向下传递 GHI 信号
        if self.norm is not None:
            x = self.norm(x)
        return x


class Model(nn.Module):
    """
    Paper Name: Physics-Guided MoE-iTransformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 提取 GHI 的特征索引 (默认 2)
        self.ghi_index = 2

        # 变量反转的投影层
        self.projector = nn.Linear(self.seq_len, configs.d_model, bias=True)

        # 初始化带 MoE 的深度编码器
        self.encoder = MoE_Encoder(configs)

        # 最终预测头
        self.projection = nn.Linear(configs.d_model, self.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. 切出 GHI 并计算先验 (注意：此时已经是 Z-score 标准化后的数据)
        ghi_seq_z = x_enc[:, :, self.ghi_index]  # [Batch, 96]
        ghi_mean_z = torch.mean(ghi_seq_z, dim=1)  # [Batch]

        # 2. iTransformer 的序列规范化与变量反转
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 反转并投影: [B, L, C] -> [B, C, L] -> [B, C, d_model]
        enc_out = x_enc.permute(0, 2, 1)
        enc_out = self.projector(enc_out)

        # 3. 传入深层 MoE 编码器
        enc_out = self.encoder(enc_out, ghi_mean_z)

        # 4. 反向规范化并输出预测
        dec_out = self.projection(enc_out).permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out