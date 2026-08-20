import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    单个专家网络：标准的双层前馈神经网络 (FFN)
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class PhysicsGuidedMoE(nn.Module):
    """
    基于 GHI 物理硬约束引导的混合专家层
    """

    def __init__(self, d_model=128, num_experts=3, d_ff=256):
        super().__init__()
        self.num_experts = num_experts

        # 定义 3 个专家：0-弱光专家, 1-中光专家, 2-强光专家
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

        # 门控网络：输入为单维度的 GHI 统计值，输出 3 个专家的初始分配倾向
        self.gate = nn.Linear(1, num_experts)

    def forward(self, x, ghi_mean):
        """
        x: iTransformer 处理后的变量 Token [Batch, C, d_model]
        ghi_mean: 历史窗口内的 GHI 均值或最大值 [Batch]
        """
        B, C, D = x.shape

        # 1. 计算初始门控得分 (Logits)
        # 将 [B] 扩展为 [B, 1] 以匹配 Linear 层输入
        gate_logits = self.gate(ghi_mean.unsqueeze(-1))  # 形状: [B, 3]

        # 2. 注入物理硬约束掩码 (Hard Masking)
        # 注意：这里的阈值 (150, 600) 需根据你实际数据集 GHI 的统计分布进行微调
        mask = torch.zeros_like(gate_logits)

        # 规则 A：如果历史 GHI 极低（如阴雨/夜间），强行屏蔽强光专家（索引 2）
        mask[ghi_mean < 150, 2] = -1e9

        # 规则 B：如果历史 GHI 极高（如连续晴天），强行屏蔽弱光专家（索引 0）
        mask[ghi_mean > 600, 0] = -1e9

        # 将掩码叠加到 logits 上，被屏蔽的位置在 Softmax 后概率严格为 0
        gate_logits = gate_logits + mask
        routing_weights = F.softmax(gate_logits, dim=-1)  # 形状: [B, 3]

        # 3. Top-1 稀疏路由分配 (极大地节省 RTX 2080 显存)
        # 获取每个 Batch 样本对应的概率最大的专家索引
        top1_weights, top1_indices = torch.max(routing_weights, dim=-1)  # 形状: [B], [B]

        # 初始化输出张量
        final_output = torch.zeros_like(x)

        # 遍历 3 个专家，仅对分配到该专家的样本进行前向计算
        for i, expert in enumerate(self.experts):
            # 找到当前 Batch 中被分配给专家 i 的样本索引
            batch_idx = (top1_indices == i).nonzero(as_tuple=True)[0]

            if batch_idx.numel() > 0:
                # 提取对应的特征切片: [选中的样本数, C, d_model]
                expert_input = x[batch_idx]

                # 专家进行计算，并乘以对应的门控权重
                expert_output = expert(expert_input) * top1_weights[batch_idx].view(-1, 1, 1)

                # 将计算结果原路填回输出张量
                final_output[batch_idx] = expert_output

        return final_output