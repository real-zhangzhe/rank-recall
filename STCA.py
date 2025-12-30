"""
The unofficial implementation of Stacked Target-to-History Cross Attention(STCA) module of paper "Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin"
time: 2025/12/29
author: soaprockets
"""


import tensorflow as tf 
from tensorflow.keras.layers import *


class SwiGLUFFN(Layer):
    def __init__(self, d_model, expand_ratio=4, **kwargs):
        super().__init__(**kwargs)
        d_ff = int(d_model * expand_ratio)

        self.w1 = Dense(d_ff, activation='linear', name='w1')
        self.w2 = Dense(d_ff, activation='linear', name='w2')
        self.w3 = Dense(d_model, activation='linear', name='w3')
        
        self.norm = LayerNormalization(epsilon=1e-6)
        
    def call(self,x):
        if len(x.shape)==2:
            x = x[:,None,:]
        x = self.norm(self.w3(tf.nn.silu(self.w1(x)) * self.w2(x)))
        return x

class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads

        self.query_ffn = SwiGLUFFN(d_model)
        self.hist_ffn  = SwiGLUFFN(d_model)

        self.w_q = tf.keras.layers.Dense(d_model, name='w_q')
        self.w_k = tf.keras.layers.Dense(d_model, name='w_k')
        self.w_v = tf.keras.layers.Dense(d_model, name='w_v')
        self.w_o = tf.keras.layers.Dense(d_model, name='w_o')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, query, history):
        batch_size = tf.shape(query)[0]

        query = self.query_ffn(query)            # (B,1,d_model)
        history = self.hist_ffn(history)         # (B,T,d_model)

        Q = self.w_q(query)                      # (B,1,d_model)
        K = self.w_k(history)                    # (B,T,d_model)
        V = self.w_v(history)                    # (B,T,d_model)

        Q = tf.reshape(Q, (batch_size, 1, self.num_heads, self.d_h))
        K = tf.reshape(K, (batch_size, -1, self.num_heads, self.d_h))
        V = tf.reshape(V, (batch_size, -1, self.num_heads, self.d_h))

        Q = tf.transpose(Q, [0, 2, 1, 3])        # (B,H,1,d_h)
        K = tf.transpose(K, [0, 2, 1, 3])        # (B,H,T,d_h)
        V = tf.transpose(V, [0, 2, 1, 3])        # (B,H,T,d_h)  <-- 你原来这里漏了

        scale = tf.math.sqrt(tf.cast(self.d_h, tf.float32))
        alpha = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / scale, axis=-1)  # (B,H,1,T)

        out = tf.matmul(alpha, V)                # (B,H,1,d_h)
        out = tf.transpose(out, [0, 2, 1, 3])    # (B,1,H,d_h)
        out = tf.reshape(out, (batch_size, 1, self.d_model))  # (B,1,d_model)

        out = self.w_o(out)
        out = self.norm(out)
        return out          # (B,1,d_model)
    
    

    
class StackedMultiHeadCrossAttention(Layer):
    def __init__(self, num_layers, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.proj_q = SwiGLUFFN(d_model)
        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(MultiHeadCrossAttention(d_model, num_heads))
        self.proj_o = SwiGLUFFN(d_model)
    def call(self,query,history):
        outputs =[self.layers_list[i](query,history) for i in range(len(self.layers_list))]
        outputs += [self.proj_q(query)] # target-aware
        outputs = tf.reshape(tf.concat(outputs,axis=1),[-1,(self.num_layers+1)* self.d_model])  # (B,num_layers+1,d_model) -> (B,(num_layers+1)*d_model)
        output = self.proj_o(outputs) # 
        
        return tf.squeeze(output,axis=1)

class HistoryItemLayer(Layer):
    def __init__(self,his_length,**kwargs):
        super().__init__(**kwargs)
        self.his_length = his_length
        
    def call(self,x):
        his_item =[]
        for i in range(self.his_length):
            item_split=[]
            for item in x:
                item_split.append(item[:,i,:])
            item_split = tf.concat(item_split,axis=-1) # (B,feature_dim)
            his_item.append(item_split)
        his_item = tf.stack(his_item,axis=1)  # (B,his_length,feature_dim)
        return his_item
             
        
