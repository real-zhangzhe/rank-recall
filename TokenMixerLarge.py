import tensorflow as tf
from tensorflow.keras.layers import *


class SemanticGroupTokenizer(Layer):
    """
    该模块只是演示将所有分组的embedding特征使用独立的dnn/mlp进行映射，实际特征前处理时已经做好了分组和压缩，
    这个模块的代码仅为参考，后续tokenmixer-large不会直接使用这个模块。
    """

    def __init__(self, d_model, n_groups, **kwargs):
        super().__init__(**kwargs)
        self.n_groups = n_groups
        self.mlp = [Dense(d_model) for _ in range(self.n_groups)]

    def call(self, x):  # x : list of [B, in_dim]
        outputs = []
        for i, layer in enumerate(self.mlp):
            outputs.append(layer(x[i]))
        return tf.stack(outputs)  # (B, n_groups, d_model)


class GlobalTokenizer(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.mlp = Dense(d_model)

    def call(self, x):  # x : [B, in_dim]
        return self.mlp(x)  # (B, d_model)


class RMSLayerNorm(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            "scale", shape=(input_shape[-1],), initializer="ones"
        )

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


class TokenMixing(Layer):
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


class SwiGLU(Layer):
    def __init__(self, num_D, expansion_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.fc_up = Dense(num_D * expansion_ratio)
        self.fc_gate = Dense(num_D * expansion_ratio, activation="swish")
        self.fc_down = Dense(num_D)

    def call(self, x):
        x = self.fc_down(self.fc_up(x) * self.fc_gate(x))
        return x


class PSwiGLU(Layer):
    def __init__(self, num_T, num_D, expansion_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.mlp = [SwiGLU(num_D, expansion_ratio) for _ in range(num_T)]

    def call(self, x):
        outputs = []
        for i in range(self.num_T):
            h = x[:, i, :]
            outputs.append(self.mlp[i](h))
        return tf.stack(outputs, axis=1)  # (B,T,D)


class TokenMixerLargeBlock(Layer):
    def __init__(self, num_T, num_D, num_H, expansion_ratio, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = RMSLayerNorm()
        self.norm2 = RMSLayerNorm()
        self.fc1 = PSwiGLU(num_T, num_D, expansion_ratio)
        self.fc2 = PSwiGLU(num_T, num_D, expansion_ratio)
        self.token_mixer = TokenMixing(num_T, num_D, num_H)
        self.token_revert = TokenMixing(
            num_T, num_H, num_D
        )  # 与token_mixer结构相同，用于将token_mixer的输出还原回原始维度, num_D和num_H交换位置

    def call(self, x):
        mixed_x = self.token_mixer(x)
        x = self.norm1(mixed_x + self.fc1(mixed_x))
        revert_x = self.token_revert(x)
        x = self.norm2(revert_x + self.fc2(revert_x))
        return x


class TokenMixerLarge(Layer):
    def __init__(self, num_blocks, num_T, num_D, num_H, expansion_ratio, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            TokenMixerLargeBlock(num_T, num_D, num_H, expansion_ratio)
            for _ in range(num_blocks)
        ]

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x
