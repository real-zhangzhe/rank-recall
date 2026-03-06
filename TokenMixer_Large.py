import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        # 对应 nn.Parameter(torch.ones(dim))
        self.scale = self.add_weight(
            shape=(dim,), initializer="ones", trainable=True, name="scale"
        )

    def call(self, x):
        # norm = x.pow(2).mean(-1, keepdim=True)
        norm = tf.reduce_mean(tf.math.pow(x, 2), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(norm + self.eps)
        return self.scale * x


class PertokenSwiGLU(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_mult=4, down_scale=0.01, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = tf.keras.layers.Dense(hidden_dim)
        self.fc_gate = tf.keras.layers.Dense(hidden_dim)

        # down-matrix small init: 对应 xavier_uniform_ 配合 gain
        # Xavier uniform 的极值是 sqrt(6 / (fan_in + fan_out))
        limit = tf.math.sqrt(6.0 / (hidden_dim + dim)) * down_scale
        init = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        self.fc_down = tf.keras.layers.Dense(dim, kernel_initializer=init)

    def call(self, x):
        up = self.fc_up(x)
        gate_x = self.fc_gate(x)
        # 等效于 torch.sigmoid(gate_x) * gate_x
        gate = tf.nn.swish(gate_x)
        return self.fc_down(up * gate)


class SparsePertokenMoE(tf.keras.layers.Layer):
    def __init__(self, dim, num_experts=4, top_k=2, hidden_mult=4, alpha=2.0, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = float(alpha)

        self.router = tf.keras.layers.Dense(num_experts)

        # experts 列表
        self.experts = [
            PertokenSwiGLU(dim, hidden_mult) for _ in range(num_experts - 1)
        ]
        # shared expert
        self.shared_expert = PertokenSwiGLU(dim, hidden_mult)

    def call(self, x):
        logits = self.router(x)
        probs = tf.nn.softmax(logits, axis=-1)

        # 取 Top-K
        topk_vals, topk_idx = tf.math.top_k(probs, k=self.top_k)

        output = tf.zeros_like(x)

        # PyTorch 原代码只循环了 top_k - 1 次
        for i in range(self.top_k - 1):
            expert_idx = topk_idx[..., i]
            expert_prob = tf.expand_dims(topk_vals[..., i], axis=-1)

            expert_out = tf.zeros_like(x)
            for j, expert in enumerate(self.experts):
                # 找出哪些 token 分配给了当前专家 j
                mask = tf.equal(expert_idx, j)
                mask_float = tf.cast(mask, x.dtype)
                mask_expanded = tf.expand_dims(mask_float, axis=-1)

                # TF 图模式下更稳定、且无动态切片报错的写法：
                # 计算全部结果，但只保留 mask 位置的输出（不影响梯度和实际效果）
                expert_out += expert(x) * mask_expanded

            output += self.alpha * expert_prob * expert_out

        # shared expert always activated
        output += self.shared_expert(x)

        return output


class MixingReverting(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.mixing = PertokenSwiGLU(dim)
        self.reverting = PertokenSwiGLU(dim)

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = x.shape[-1]
        H = self.num_heads
        d = D // H

        # -------- Mixing --------
        x_norm = self.norm1(x)

        # view -> reshape
        x_split = tf.reshape(x_norm, [B, T, H, d])
        # permute -> transpose
        x_split = tf.transpose(x_split, [2, 0, 1, 3])
        # view -> reshape
        x_split = tf.reshape(x_split, [H, B, T * d])

        x_mixed = self.mixing(x_split)

        # -------- Reverting --------
        x_rev = tf.reshape(x_mixed, [H, B, T, d])
        x_rev = tf.transpose(x_rev, [1, 2, 0, 3])
        x_rev = tf.reshape(x_rev, [B, T, D])

        x_out = self.norm2(x + self.reverting(x_rev))

        return x_out


class TokenMixerLargeBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, num_experts=4, top_k=2, hidden_mult=4, **kwargs):
        super().__init__(**kwargs)
        self.mr = MixingReverting(dim, num_heads)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult)

    def call(self, x):
        x = self.mr(x)
        x = x + self.moe(self.norm(x))
        return x


class SemanticTokenizer(tf.keras.layers.Layer):
    def __init__(self, group_dims, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlps = []
        for dims in group_dims:
            self.mlps.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(model_dim),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(model_dim),
                    ]
                )
            )

        self.global_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(model_dim),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(model_dim),
            ]
        )

    def call(self, groups):
        tokens = []

        for group, mlp in zip(groups, self.mlps):
            # concatenate
            concat = tf.concat(group, axis=-1)
            tokens.append(mlp(concat))

        # stack
        stacked = tf.stack(tokens, axis=1)

        # global mlp
        stacked_flat = tf.reshape(stacked, [tf.shape(stacked)[0], -1])
        global_token = tf.expand_dims(self.global_mlp(stacked_flat), axis=1)

        return tf.concat([global_token, stacked], axis=1)


class TokenMixerLarge(tf.keras.Model):
    def __init__(
        self,
        group_dims,
        model_dim=256,
        depth=6,
        num_heads=8,
        num_experts=4,
        top_k=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tokenizer = SemanticTokenizer(group_dims, model_dim)

        self.blocks = [
            TokenMixerLargeBlock(model_dim, num_heads, num_experts, top_k)
            for _ in range(depth)
        ]

        self.head = tf.keras.layers.Dense(1)

    def call(self, groups):
        x = self.tokenizer(groups)

        residual_cache = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            # interval residual every 2 layers
            if i % 2 == 1:
                x = x + residual_cache[-1]
                residual_cache = []

            residual_cache.append(x)

        # pooled
        pooled = tf.reduce_mean(x, axis=1)
        return self.head(pooled)


# ==========================================
# 测试代码运行 (TensorFlow 版本)
# ==========================================
if __name__ == "__main__":
    B = 8
    group_dims = [[16, 8], [12, 12], [10, 6]]

    # 使用 tf.random.normal 模拟输入
    groups = [[tf.random.normal((B, d)) for d in dims] for dims in group_dims]

    # 初始化模型
    model = TokenMixerLarge(
        group_dims, model_dim=128, depth=4, num_heads=4, num_experts=4
    )

    # 跑一次前向传播
    out = model(groups)
    print("输出维度:", out.shape)  # 预期输出: (8, 1)
