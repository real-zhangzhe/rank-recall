import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense




class SequenceTokenizer(Layer):
    """
    输入特征 [x1,x2,...,xn]是用户点击过的一个item的特征属性， 维度是[B,Seq_len,in_dim],所有特征拼接表示一个完整的历史点击行为
    只要保证最后一个行为序列Si 的维度映射成[B,L,d_model]即可
    这里提供几种常用的tokenizer实现方式
    """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pro_j = Dense(d_model)
    def call(self, x):  # x : list of [B, in_dim] 
        x = tf.concat(x,axis=-1) # [B, seq_len, D]
        return self.pro_j(x)  # [B, seq_len, d_model]
        

class MultiBehaviorTokenizer(Layer):
    def __init__(self,d_model, n_behaviors, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_behaviors = n_behaviors
        self.seq = self.add_weight("seq", (n_behaviors-1,d_model),
                                   initializer="glorot_uniform")
    def call(self, x):
        B = tf.shape(x[0])[0]
        out = x[0]
        for i in range(1, self.n_behaviors):
            sep = tf.tile(self.seq[i-1][None, None, :], [B, 1, 1])  # [B,1,d]
            out = tf.concat([out, sep, x[i]], axis=1)
        return out  
        
     
class AutoSplitTokenizer(Layer):
    def __init__(self, num_T, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.d_model = d_model
        self.proj = Dense(num_T * d_model)

    def call(self, x):  # x: [B, in_dim]
        x = self.proj(x)
        return tf.reshape(x, [-1, self.num_T, self.d_model])  # [B, num_T, d_model]


class GroupWiseTokenizer(Layer):
    def __init__(self, num_T, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.proj = [Dense(d_model) for _ in range(num_T)]

    def call(self, x):  # x: [B, in_dim],  in_dim % num_T == 0
        parts = tf.split(x, self.num_T, axis=-1)                 # list of [B, in_dim/num_T]
        tokens = [p(parts[i]) for i, p in enumerate(self.proj)]  # each -> [B, d_model]
        return tf.stack(tokens, axis=1)    
# ---------------- RMSNorm ----------------
class RMSLayerNorm(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=(input_shape[-1],), initializer="ones")

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


# ---------------- Mixed FFN (tail LNS token-specific) ----------------
class MixedFFN(Layer):
    # 尾部 min(LNS, T) token 用 token-specific，其余 shared
    def __init__(self, d_model, d_ff, LNS, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.LNS = LNS
        self.W1S, self.W2S = Dense(d_ff), Dense(d_model)

        init = tf.keras.initializers.GlorotUniform()
        self.W1NS = self.add_weight("W1NS", (LNS, d_model, d_ff), initializer=init)
        self.W2NS = self.add_weight("W2NS", (LNS, d_ff, d_model), initializer=init)

        self.act = tf.keras.activations.get(activation)

    def call(self, x):
        T = tf.shape(x)[1]
        t = tf.minimum(T, self.LNS)   # tail token-specific count
        s = T - t                     # shared count

        yS = self.W2S(self.act(self.W1S(x[:, :s])))  # [B,s,D]

        xT = x[:, s:]                                # [B,t,D]
        W1 = self.W1NS[-t:]
        W2 = self.W2NS[-t:]
        h  = self.act(tf.einsum("btd,tde->bte", xT, W1))
        yT = tf.einsum("btd,tde->bte", h, W2) # [B,t,D]

        return tf.concat([yS, yT], axis=1)


# ---------------- Pyramid Mixed Causal Attention (Eq.14 strict) ----------------
class PyramidMixedCausalAttention(Layer):
    """
    Eq.(14) strict:
      - Q from tail set (length Lq)
      - K/V from full sequence (length L)
      - output keeps only tail (length Lq)
    Mixed parameterization:
      - tail min(L, LNS) tokens use token-specific
      - earlier tokens use shared
    """
    def __init__(self, d_model, num_heads, LNS, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.D, self.H, self.dh = d_model, num_heads, d_model // num_heads
        self.LNS = LNS

        self.WqS = Dense(d_model, use_bias=False)
        self.WkS = Dense(d_model, use_bias=False)
        self.WvS = Dense(d_model, use_bias=False)

        init = tf.keras.initializers.GlorotUniform()
        self.WqNS = self.add_weight("WqNS", (LNS, d_model, d_model), initializer=init)
        self.WkNS = self.add_weight("WkNS", (LNS, d_model, d_model), initializer=init)
        self.WvNS = self.add_weight("WvNS", (LNS, d_model, d_model), initializer=init)

        self.Wo = Dense(d_model, use_bias=False)

    def _mh(self, x):  # [B,T,D] -> [B,H,T,dh]
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, [b, t, self.H, self.dh]), [0, 2, 1, 3])

    def _unmh(self, x):  # [B,H,T,dh] -> [B,T,D]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.D])

    def call(self, x, Lq):
        L = tf.shape(x)[1]
        t = tf.minimum(L, self.LNS)   # tail token-specific
        s = L - t                     # head shared

        xS, xT = x[:, :s], x[:, s:]

        Q = tf.concat([self.WqS(xS), tf.einsum("btd,tde->bte", xT, self.WqNS[-t:])], 1)
        K = tf.concat([self.WkS(xS), tf.einsum("btd,tde->bte", xT, self.WkNS[-t:])], 1)
        V = tf.concat([self.WvS(xS), tf.einsum("btd,tde->bte", xT, self.WvNS[-t:])], 1)

        Q = Q[:, -Lq:]  # only tail queries (all tokens in Q will be updated)

        Qh, Kh, Vh = self._mh(Q), self._mh(K), self._mh(V)
        logits = tf.matmul(Qh, Kh, transpose_b=True) * (tf.cast(self.dh, tf.float32) ** -0.5)

        # causal mask for tail queries: absolute indices [L-Lq .. L-1]
        q = tf.range(L - Lq, L)[:, None]
        k = tf.range(L)[None, :]
        logits += tf.cast(k > q, tf.float32)[None, None] * (-1e9)

        out = tf.matmul(tf.nn.softmax(logits, -1), Vh)  # [B,H,Lq,dh]
        return self.Wo(self._unmh(out))                 # [B,Lq,D]


# ---------------- OneTrans Block (auto Lq=L-1) ----------------
class OneTransBlock(Layer):
    def __init__(self, d_model, num_heads, d_ff, LNS, ln_eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.ln1, self.ln2 = RMSLayerNorm(ln_eps), RMSLayerNorm(ln_eps)
        self.mha = PyramidMixedCausalAttention(d_model, num_heads, LNS)
        self.ffn = MixedFFN(d_model, d_ff, LNS)

    def call(self, x, Lq):
        z = self.mha(self.ln1(x), Lq) + x[:, -Lq:]   # residual对齐尾部
        return self.ffn(self.ln2(z)) + z


# ---------------- Stack: compress S for LS layers ----------------

class OneTrans(Layer):
    def __init__(self, LS, d_model, num_heads, d_ff, LNS, n_task, **kwargs):
        super().__init__(**kwargs)
        self.ctr_dense = Dense(d_model)
        self.cvr_dense = Dense(d_model)
        self.Lq_list = list(range(LS+LNS, LNS, -4))  # LS+LNS .. LNS+1
        self.Lq_list.append(LNS)      
        self.blocks = [
            OneTransBlock(d_model, num_heads, d_ff, LNS=LNS)
            for _ in range(len(self.Lq_list))
        ]# finally LNS. 
        self.task_tower = Dense(n_task)

    def call(self, x):
        h = x
        for blk,Lq_py in zip(self.blocks, self.Lq_list):
            h = blk(h, Lq_py)   
        h = tf.transpose(h, [0,2,1])  # [B,D,LNS]
        h = self.task_tower(h)  # [B,D,n_task]
        h = tf.transpose(h, [0,2,1])  # [B,n_task,D]
        return h  # [B, n_task, D]


# ---------------- Test ----------------
def test_onetrans():
    tf.random.set_seed(0)

    B = 2
    LS, LNS = 4, 2
    D_MODEL = 32
    NUM_HEAD = 4
    D_FF = 64


    model = OneTrans(LS=LS, d_model=D_MODEL, num_heads=NUM_HEAD, d_ff=D_FF, LNS=LNS,n_task=2)
    inputs = tf.random.normal([B, LS + LNS, D_MODEL])
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    
if __name__ == "__main__":
    test_onetrans()
