import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Custom Layers ---


class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(self.dim,), initializer="ones", trainable=True
        )

    def call(self, x):
        # norm = mean(x^2)
        norm = tf.reduce_mean(tf.pow(x, 2), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(norm + self.eps)
        return self.scale * x


class PertokenSwiGLU(layers.Layer):
    def __init__(self, dim, hidden_mult=4, down_scale=0.01, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = layers.Dense(hidden_dim)
        self.fc_gate = layers.Dense(hidden_dim)
        self.fc_down = layers.Dense(
            dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=down_scale, mode="fan_avg", distribution="uniform"
            ),
        )

    def call(self, x):
        up = self.fc_up(x)
        gate_logits = self.fc_gate(x)
        # Swish = x * sigmoid(x)
        gate = tf.nn.sigmoid(gate_logits) * gate_logits
        return self.fc_down(up * gate)


class SparsePertokenMoE(layers.Layer):
    def __init__(self, dim, num_experts=4, top_k=2, hidden_mult=4, alpha=2.0, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

        self.router = layers.Dense(num_experts)
        # In TF 2.4, we just use a list of layers
        self.experts = [
            PertokenSwiGLU(dim, hidden_mult) for _ in range(num_experts - 1)
        ]
        self.shared_expert = PertokenSwiGLU(dim, hidden_mult)

    def call(self, x):
        # Shape: (B, T, D)
        logits = self.router(x)
        probs = tf.nn.softmax(logits, axis=-1)

        # Get top-k
        topk_vals, topk_idx = tf.math.top_k(probs, k=self.top_k)

        # We initialize output as zeros
        output = tf.zeros_like(x)

        # Loop over top-k (excluding the last slot if strictly following PyTorch logic)
        for i in range(self.top_k - 1):
            expert_prob = tf.expand_dims(topk_vals[..., i], axis=-1)
            indices = topk_idx[..., i]

            expert_outputs_sum = tf.zeros_like(x)
            for j, expert in enumerate(self.experts):
                # Create mask for which tokens go to which expert
                mask = tf.cast(tf.equal(indices, j), dtype=x.dtype)
                mask = tf.expand_dims(mask, axis=-1)

                # Apply expert to all then mask (more TF friendly than boolean indexing)
                exp_out = expert(x)
                expert_outputs_sum += exp_out * mask

            output += self.alpha * expert_prob * expert_outputs_sum

        # Shared expert always activated
        output += self.shared_expert(x)
        return output


class MixingReverting(layers.Layer):
    def __init__(self, dim, num_heads, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dim = dim
        self.num_tokens = num_tokens

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        d = dim // num_heads
        mix_dim = num_tokens * d

        self.mixing = PertokenSwiGLU(mix_dim)
        self.reverting = PertokenSwiGLU(dim)

    def call(self, x):
        # B, T, D
        batch_size = tf.shape(x)[0]
        H = self.num_heads
        d = self.dim // H
        T = self.num_tokens

        x_norm = self.norm1(x)

        # Reshape for multi-head mixing: [B, T, H, d]
        x_split = tf.reshape(x_norm, (batch_size, T, H, d))
        # Permute to: [H, B, T, d]
        x_split = tf.transpose(x_split, perm=[2, 0, 1, 3])
        # Flatten T and d: [H, B, T*d]
        x_split = tf.reshape(x_split, (H, batch_size, T * d))

        x_mixed = self.mixing(x_split)

        # Reverse: [H, B, T, d]
        x_rev = tf.reshape(x_mixed, (H, batch_size, T, d))
        # [B, T, H, d]
        x_rev = tf.transpose(x_rev, perm=[1, 2, 0, 3])
        # [B, T, D]
        x_rev = tf.reshape(x_rev, (batch_size, T, self.dim))

        return self.norm2(x + self.reverting(x_rev))


class TokenMixerLargeBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        num_tokens,
        num_experts=4,
        top_k=2,
        hidden_mult=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mr = MixingReverting(dim, num_heads, num_tokens)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult)

    def call(self, x):
        x = self.mr(x)
        x = x + self.moe(self.norm(x))
        return x


class SemanticTokenizer(layers.Layer):
    def __init__(self, group_dims, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlps = []
        for dims in group_dims:
            self.mlps.append(
                tf.keras.Sequential(
                    [
                        layers.Dense(model_dim, activation="relu"),
                        layers.Dense(model_dim),
                    ]
                )
            )

        self.global_mlp = tf.keras.Sequential(
            [layers.Dense(model_dim, activation="relu"), layers.Dense(model_dim)]
        )

    def call(self, groups):
        # groups is a list of lists of tensors
        tokens = []
        for group_tensors, mlp in zip(groups, self.mlps):
            # Concatenate tensors in each group
            concat = tf.concat(group_tensors, axis=-1)
            tokens.append(mlp(concat))

        # Stack into [B, T-1, D]
        stacked = tf.stack(tokens, axis=1)

        # Global token: Flatten stacked and pass through global_mlp
        batch_size = tf.shape(stacked)[0]
        flattened = tf.reshape(stacked, (batch_size, -1))
        global_token = self.global_mlp(flattened)
        global_token = tf.expand_dims(global_token, axis=1)

        # Concatenate global token with group tokens
        return tf.concat([global_token, stacked], axis=1)


class TokenMixerLarge(Model):
    def __init__(
        self,
        group_dims,
        model_dim=256,
        depth=6,
        num_heads=8,
        num_experts=4,
        top_k=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = SemanticTokenizer(group_dims, model_dim)
        num_tokens = len(group_dims) + 1

        self.blocks = [
            TokenMixerLargeBlock(model_dim, num_heads, num_tokens, num_experts, top_k)
            for _ in range(depth)
        ]
        self.head = layers.Dense(1)

    def call(self, groups):
        x = self.tokenizer(groups)

        residual_cache = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            if i % 2 == 1:
                x = x + residual_cache[-1]
                residual_cache = []

            residual_cache.append(x)

        pooled = tf.reduce_mean(x, axis=1)
        return self.head(pooled)


# --- Testing Code ---

B = 8
# Mock input dimensions
group_dims = [[128] * 26, [128] * 13]
# Create mock tensors
groups = [[tf.random.normal((B, d)) for d in dims] for dims in group_dims]

# Initialize and run
model = TokenMixerLarge(group_dims, model_dim=128, depth=4, num_heads=4, num_experts=4)
out = model(groups)

print(f"Output shape: {out.shape}")  # Should be (8, 1)
