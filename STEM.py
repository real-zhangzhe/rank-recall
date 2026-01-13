import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Flatten
from tensorflow.keras import regularizers


class StemLayer(tf.keras.layers.Layer):
    def __init__(self, n_task, n_experts, expert_dim, n_expert_share, dnn_reg_l2=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.n_task = n_task
        self.expert_dim = expert_dim
        self.n_experts = n_experts

        self.E_layer = []
        for i in range(n_task):
            sub_exp = []
            for _ in range(n_experts[i]):
                sub_exp.append([
                    Dense(expert_dim, activation="swish",
                          kernel_regularizer=regularizers.l2(dnn_reg_l2)),
                    Dropout(0.3)
                ])
            self.E_layer.append(sub_exp)

        self.share_layer = []
        for _ in range(n_expert_share):
            self.share_layer.append([
                Dense(expert_dim, activation="swish",
                      kernel_regularizer=regularizers.l2(dnn_reg_l2)),
                Dropout(0.3)
            ])

        self.gate_layers = [
            Dense(n_expert_share + sum(n_experts),
                  activation="softmax",
                  kernel_regularizer=regularizers.l2(dnn_reg_l2))
            for _ in range(n_task + 1)
        ]

    def call(self, x):
        xs = tf.split(x, self.n_task + 1, axis=-1)

        E_net = []
        for i, sub_exp in enumerate(self.E_layer):
            E_net.append([drop(dense(xs[i])) for dense, drop in sub_exp])

        share_net = [drop(dense(xs[-1])) for dense, drop in self.share_layer]

        towers = []
        for i in range(self.n_task):
            g = self.gate_layers[i](xs[i])
            g = tf.expand_dims(g, axis=-1)

            experts = []
            experts.extend(share_net)
            for j in range(self.n_task):
                experts.extend(E_net[j])

            experts = Concatenate(axis=1)([e[:, tf.newaxis, :] for e in experts])
            tower = tf.matmul(experts, g, transpose_a=True)
            towers.append(Flatten()(tower))

        return towers
