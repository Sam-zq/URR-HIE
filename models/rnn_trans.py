import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding2


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        self.patch_embedding = PatchEmbedding2(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.temporal_gru = nn.GRU(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=2,
            batch_first=True,
            dropout=configs.dropout
        )

        self.ar_gru_cell = nn.GRUCell(configs.d_model, configs.d_model)

        self.ar_mlp = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

        self.feedback_embed = nn.Linear(1, configs.d_model)

        self.variate_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # patching: [B, N, L] → [B×N, patch_num, d_model]
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # temporal GRU: 建模 patch 间时序
        gru_out, h_n = self.temporal_gru(enc_out)

        # 自回归循环: h → MLP → pred_step → feedback → GRUCell → next_h
        h_ar = h_n[-1]  # [B×N, d_model]
        ar_outputs = []
        for _ in range(self.pred_len):
            h_2d = h_ar.reshape(-1, n_vars, h_ar.shape[-1])  # [B, N, d_model]
            pred_step = self.ar_mlp(h_2d)  # [B, N, 1]
            ar_outputs.append(pred_step)
            feedback = self.feedback_embed(pred_step)  # [B, N, d_model]
            feedback = feedback.reshape(-1, feedback.shape[-1])  # [B×N, d_model]
            h_ar = self.ar_gru_cell(feedback, h_ar)  # [B×N, d_model]

        ar_outputs_tensor = torch.cat(ar_outputs, dim=-1).permute(0, 2, 1)
        ar_outputs_tensor = ar_outputs_tensor * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        ar_outputs_tensor = ar_outputs_tensor + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # AR 最终隐状态 → 变量注意力
        h_final = h_ar.reshape(-1, n_vars, h_ar.shape[-1])  # [B, N, d_model]
        var_out, _ = self.variate_encoder(h_final, attn_mask=None)  # [B, N, d_model]

        # 投影输出
        dec_out = self.projection(var_out).permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]

        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, ar_outputs_tensor

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out, ar_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], ar_out[:, -self.pred_len:, :]
        return None, None
