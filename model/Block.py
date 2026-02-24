import torch
import torch.nn as nn
import torch.nn.functional as F

class IntegralSelfAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, num_signals=8):
        super().__init__()
        self.S = num_signals
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_h = self.head_dim // self.S
        
        # 增加 LayerNorm 稳定输入分布
        self.norm = nn.LayerNorm(d_model)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # 1. Pre-norm
        x = self.norm(x)
        
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.S, self.d_h).permute(0, 2, 3, 1, 4)
        k = self.k_proj(x).view(B, N, self.num_heads, self.S, self.d_h).permute(0, 2, 3, 1, 4)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. 计算点积并缩放
        # 注意：如果 d_h 太小，可以尝试手动调大这个缩放因子，例如使用 8.0
        scaling = self.d_h ** 0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) / scaling
        
        # 3. 积分 (平均)
        avg_logits = logits.mean(dim=2) # [B, num_heads, N, N]
        
        # 4. 数值稳定的 Mask
        if mask is not None:
            # mask shape: [B, N] -> [B, 1, 1, N]
            mask = mask.unsqueeze(1).unsqueeze(2)
            # 使用足够大的负数而不是 -inf，防止整行全被 mask 导致的 NaN
            avg_logits = avg_logits.masked_fill(mask == 0, -1e4) 
        
        # 5. 计算 Softmax 之前减去最大值（数值稳定性技巧，PyTorch 默认已有，但手动确保更安全）
        attn_weights = F.softmax(avg_logits, dim=-1)
        
        # 6. 处理可能的残差全零导致的 NaN (冗余保护)
        attn_weights = torch.nan_to_num(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        return self.out_proj(out)