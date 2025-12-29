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
        original_shape = tf.shape(x)
        if tf.rank(x) ==2:
            x = tf.expand_dims(x, axis=1)  # (B,D) -> (B,1,D)
        x = self.norm(self.w3(tf.nn.silu(self.w1(x)) * self.w2(x)))
        if tf.rank(x) == 3 and original_shape[1] == 1:
            x = tf.squeeze(x, axis=1)  # (B,1,D) -> (B,D)
        return x

class MultiHeadCrossAttention(Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model//num_heads
        
        self.query = SwiGLUFFN(d_model)
        self.history = SwiGLUFFN(d_model)
        
        self.w_q = Dense(self.d_model*self.d_h, activation='linear', name='w_q')
        self.w_k = Dense(self.d_model*self.d_h, activation='linear', name='w_k')
        self.w_v = Dense(self.d_model*self.d_h, activation='linear', name='w_v')
        
        self.w_o = Dense(d_model, activation='linear', name='w_o')
        self.norm = LayerNormalization(epsilon=1e-6)
        
    def call(self,query,history):
        batch_size = tf.shape(query)[0]
        query = self.query(query)
        history = self.history(history)
        
        Q = self.w_q(query)
        K = self.w_k(history)
        V = self.w_v(history)
        
        Q = tf.reshape(Q,(batch_size,1,self.num_heads,self.d_h))
        K = tf.reshape(K,(batch_size,-1,self.num_heads,self.d_h))
        V = tf.reshape(V,(batch_size,-1,self.num_heads,self.d_h))
        
        Q = tf.transpose(Q,perm=[0,2,1,3])  # (B,H,1,d_h)
        K = tf.transpose(K,perm=[0,2,1,3])  #
        K = tf.transpose(K,perm=[0,2,1,3])
        
        alpha = tf.nn.softmax(tf.matmul(Q,K,transpose_b=True)/tf.math.sqrt(tf.cast(self.d_h,tf.float32)),axis=-1)  # (B,H,1,seq_len)
        output = tf.matmul(alpha,V)  # (B,H,1,d_h)
        output = tf.transpose(output,perm=[0,2,1,3])  # (B,1,H,d_h)
        output = tf.reshape(output,(batch_size,1,self.d_model))  # (B,1,d_model)
        output = self.w_o(output) # (B,1,d_model)
        output = self.norm(output)
        output = tf.squeeze(output,axis=1)  # (B,d_model)
        return output
    
    

    
class StackedMultiHeadCrossAttention(Layer):
    def __init__(self, num_layers, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(MultiHeadCrossAttention(d_model, num_heads))
        self.proj = SwiGLUFFN(d_model)
    def call(self,query,history):
        outputs =[self.layers_list[i](query,history) for i in range(len(self.layers_list))]
        outputs += [query] # target-aware
        outputs = tf.stack(outputs,axis=1)  # (B,num_layers+1,d_model)
        output = self.proj(outputs) # 
        return output

