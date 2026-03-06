import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.scale * x


class PertokenSwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4, down_scale=0.01):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = nn.Linear(dim, hidden_dim)
        self.fc_gate = nn.Linear(dim, hidden_dim)
        self.fc_down = nn.Linear(hidden_dim, dim)

        # down-matrix small init
        nn.init.xavier_uniform_(self.fc_down.weight, gain=down_scale)

    def forward(self, x):
        up = self.fc_up(x)
        gate = torch.sigmoid(self.fc_gate(x)) * self.fc_gate(x)
        return self.fc_down(up * gate)


class SparsePertokenMoE(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2, hidden_mult=4, alpha=2.0):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

        self.router = nn.Linear(dim, num_experts)

        self.experts = nn.ModuleList(
            [PertokenSwiGLU(dim, hidden_mult) for _ in range(num_experts - 1)]
        )

        # shared expert
        self.shared_expert = PertokenSwiGLU(dim, hidden_mult)

    def forward(self, x):
        B, T, D = x.shape

        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        output = torch.zeros_like(x)

        for i in range(self.top_k - 1):
            expert_idx = topk_idx[..., i]
            expert_prob = topk_vals[..., i].unsqueeze(-1)

            expert_out = torch.zeros_like(x)
            for j, expert in enumerate(self.experts):
                mask = expert_idx == j
                if mask.any():
                    expert_out[mask] = expert(x[mask])

            output += self.alpha * expert_prob * expert_out

        # shared expert always activated
        output += self.shared_expert(x)

        return output


class MixingReverting(nn.Module):
    # 增加 num_tokens 参数
    def __init__(self, dim, num_heads, num_tokens):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # 计算 mixing 层实际的输入维度
        d = dim // num_heads
        mix_dim = num_tokens * d

        # mixing 处理展平后的 T * d 维度
        self.mixing = PertokenSwiGLU(mix_dim)
        # reverting 处理还原后的 D (即 dim) 维度
        self.reverting = PertokenSwiGLU(dim)

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        d = D // H

        # -------- Mixing --------
        x_norm = self.norm1(x)

        x_split = x_norm.view(B, T, H, d)
        x_split = x_split.permute(2, 0, 1, 3).contiguous()
        x_split = x_split.view(H, B, T * d)

        x_mixed = self.mixing(x_split)

        # -------- Reverting --------
        x_rev = x_mixed.view(H, B, T, d)
        x_rev = x_rev.permute(1, 2, 0, 3).contiguous()
        x_rev = x_rev.view(B, T, D)

        x_out = self.norm2(x + self.reverting(x_rev))

        return x_out


class TokenMixerLargeBlock(nn.Module):
    # 增加 num_tokens 参数
    def __init__(
        self, dim, num_heads, num_tokens, num_experts=4, top_k=2, hidden_mult=4
    ):
        super().__init__()

        # 将 num_tokens 传给 MixingReverting
        self.mr = MixingReverting(dim, num_heads, num_tokens)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult)

    def forward(self, x):
        x = self.mr(x)
        x = x + self.moe(self.norm(x))
        return x


class SemanticTokenizer(nn.Module):
    def __init__(self, group_dims, model_dim):
        super().__init__()
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(sum(dims), model_dim),
                    nn.ReLU(),
                    nn.Linear(model_dim, model_dim),
                )
                for dims in group_dims
            ]
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(len(group_dims) * model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, groups):
        # 使用 `torch.cat` 拼接每个组的特征
        tokens = []

        for group, mlp in zip(groups, self.mlps):
            concat = torch.cat(group, dim=-1)
            tokens.append(mlp(concat))

        # 通过 `stacked` 获取一个包含所有组输出的张量
        stacked = torch.stack(tokens, dim=1)

        # 添加全局token的输出
        global_token = self.global_mlp(stacked.view(stacked.size(0), -1)).unsqueeze(1)

        # 将全局token与原始tokens拼接，形成最终的输入表示
        return torch.cat([global_token, stacked], dim=1)


class TokenMixerLarge(nn.Module):
    def __init__(
        self, group_dims, model_dim=256, depth=6, num_heads=8, num_experts=4, top_k=2
    ):
        super().__init__()

        self.tokenizer = SemanticTokenizer(group_dims, model_dim)

        # 核心：在这里计算出 T 的大小 (group 数量 + 1 个 global token)
        num_tokens = len(group_dims) + 1

        self.blocks = nn.ModuleList(
            [
                # 传入 num_tokens
                TokenMixerLargeBlock(
                    model_dim, num_heads, num_tokens, num_experts, top_k
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Linear(model_dim, 1)

    def forward(self, groups):
        x = self.tokenizer(groups)

        residual_cache = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            # interval residual every 2 layers
            if i % 2 == 1:
                x = x + residual_cache[-1]
                residual_cache = []

            residual_cache.append(x)

        pooled = x.mean(dim=1)
        return self.head(pooled)


B = 8
group_dims = [[128] * 26, [128] * 13]  # 模拟两组输入，每组有多个特征，每个特征维度为128

groups = [[torch.randn(B, d) for d in dims] for dims in group_dims]

model = TokenMixerLarge(group_dims, model_dim=128, depth=4, num_heads=4, num_experts=4)

out = model(groups)
print(out.shape)  # [B, 1]
