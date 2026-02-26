import tensorflow as tf
from tensorflow.keras.layers import *


class SemanticTokenization(tf.keras.layers.Layer):
    def __init__(self, num_T, num_D, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.num_D = num_D
        self.dense_layers = [Dense(num_D, activation="linear") for _ in range(num_T)]

    def call(self, x):
        x = tf.split(x, self.num_T, axis=-1)  # (B, num_T, D/num_T)
        x = [layer(x[i]) for i, layer in enumerate(self.dense_layers)]
        return tf.stack(x, axis=1)  # (B, num_T, num_D)


class TokenMixer(Layer):
    def __init__(self, num_T, num_D, num_H, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.num_D = num_D
        self.num_H = num_H
        self.d_k = num_D // num_H

    def call(self, x):
        x = tf.reshape(
            x, (-1, self.num_T, self.num_H, self.d_k)
        )  # (B,T,D)->(B,T,H,D/H)
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B,H,T,D/H)
        x = tf.reshape(x, (-1, self.num_H, self.num_T * self.d_k))  # (B,H,T*D/H)
        return x


class PerTokenFFN(Layer):
    def __init__(self, num_T, num_D, expansion_ratio=4, **kwargs):
        super().__init__(**kwargs)

        # 每个expert的FFN
        self.experts = []
        for i in range(num_T):
            self.experts.append(
                [
                    Dense(num_D * expansion_ratio, name=f"expert_{i}_fc1"),
                    Activation("gelu"),
                    Dense(num_D, name=f"expert_{i}_fc2"),
                ]
            )

    def call(self, x):
        outputs = []
        for i, expert_layers in enumerate(self.experts):
            h = x[:, i, :]
            for layer in expert_layers:
                h = layer(h)
            outputs.append(h)

        return tf.stack(outputs, axis=1)


def gelu(x):
    return (
        0.5
        * x
        * (
            1.0
            + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3)))
        )
    )


class PerTokenSparseMoE(Layer):
    """
    Per-token Sparse MoE with ReLU routing + optional DTSI.
    """

    def __init__(
        self,
        num_T,
        num_D,
        expansion_ratio=4,
        num_experts=4,
        dropout=0.0,
        l1_coef=0.0,
        sparsity_ratio=1.0,
        use_dtsi=True,
        routing_type="relu_dtsi",
        name=None,
    ):
        super(PerTokenSparseMoE, self).__init__(name=name)
        self.num_T = int(num_T)
        self.num_D = int(num_D)
        self.expansion_ratio = int(expansion_ratio)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

    def build(self, input_shape):
        hidden_dim = self.num_D * self.expansion_ratio
        init = tf.keras.initializers.VarianceScaling(
            scale=2.0,  # TF1 默认的 factor 是 2.0 (用于 ReLU 等)
            mode="fan_in",  # 保持权重方差一致
            distribution="truncated_normal",
        )
        self.W1 = self.add_weight(
            "W1",
            [self.num_T, self.num_experts, self.num_D, hidden_dim],
            initializer=init,
        )
        self.b1 = self.add_weight(
            "b1",
            [self.num_T, self.num_experts, hidden_dim],
            initializer=tf.zeros_initializer(),
        )
        self.W2 = self.add_weight(
            "W2",
            [self.num_T, self.num_experts, hidden_dim, self.num_D],
            initializer=init,
        )
        self.b2 = self.add_weight(
            "b2",
            [self.num_T, self.num_experts, self.num_D],
            initializer=tf.zeros_initializer(),
        )
        self.gate_w_train = self.add_weight(
            "gate_w_train",
            [self.num_T, self.num_D, self.num_experts],
            initializer=init,
        )
        self.gate_b_train = self.add_weight(
            "gate_b_train",
            [self.num_T, self.num_experts],
            initializer=tf.zeros_initializer(),
        )
        if self.use_dtsi:
            self.gate_w_infer = self.add_weight(
                "gate_w_infer",
                [self.num_T, self.num_D, self.num_experts],
                initializer=init,
            )
            self.gate_b_infer = self.add_weight(
                "gate_b_infer",
                [self.num_T, self.num_experts],
                initializer=tf.zeros_initializer(),
            )
        super(PerTokenSparseMoE, self).build(input_shape)

    def _router_logits(self, x, w, b):
        # 每个 token 的路由 logits，用于专家选择。
        return tf.einsum("btd,tde->bte", x, w) + b

    def call(self, x, training=False):
        # x 形状: [B, T, D]
        # 计算每个 token 的专家输出。
        h = tf.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        expert_out = tf.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        if self.dropout and training:
            expert_out = tf.nn.dropout(expert_out, keep_prob=1.0 - self.dropout)

        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        if self.routing_type == "relu_dtsi":
            # 训练阶段使用 soft 路由以提高专家覆盖。
            gate_train = tf.nn.softmax(gate_train_logits, axis=-1)
        elif self.routing_type == "relu":
            gate_train = tf.nn.relu(gate_train_logits)
        else:
            raise ValueError("Unsupported routing_type: %s" % self.routing_type)

        if self.use_dtsi:
            # 推理阶段使用 ReLU gate 以获得稀疏激活。
            gate_infer_logits = self._router_logits(
                x, self.gate_w_infer, self.gate_b_infer
            )
            gate_infer = tf.nn.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        # 训练/推理选择不同 gate。
        gate = gate_train if training else gate_infer
        y = tf.reduce_sum(expert_out * tf.expand_dims(gate, -1), axis=2)
        return y


class RankMixerLayer(Layer):
    def __init__(self, num_T, num_D, num_H, expansion_ratio, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.token_mixer = TokenMixer(num_T, num_D, num_H)
        if use_moe:
            self.per_token_ffn = PerTokenSparseMoE(num_T, num_D, expansion_ratio)
        else:
            self.per_token_ffn = PerTokenFFN(num_T, num_D, expansion_ratio)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x):
        mixed_x = self.token_mixer(x)
        x = self.norm1(x + mixed_x)
        x = self.norm2(x + self.per_token_ffn(x))
        return x


class RankMixer(Layer):
    def __init__(self, num_layers, num_T, num_D, num_H, expansion_ratio, **kwargs):
        super().__init__(**kwargs)
        self.semantic_tokenization = SemanticTokenization(num_T, num_D)
        self.layers_list = []
        for i in range(num_layers):
            self.layers_list.append(
                RankMixerLayer(num_T, num_D, num_H, expansion_ratio, use_moe=0 == i % 2)
            )

    def call(self, x):
        x = self.semantic_tokenization(x)
        for layer in self.layers_list:
            x = layer(x)
        return tf.reduce_mean(x, axis=1)


if __name__ == "__main__":
    inputs = Input(shape=(512,))
    rankmixer = RankMixer(
        num_layers=4, num_T=8, num_D=512, num_H=8, expansion_ratio=4
    )  # to
    outputs = rankmixer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
