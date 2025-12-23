import tensorflow as tf 
from tensorflow.keras.layers import *

class SemanticTokenization(tf.keras.layers.Layer):
    def __init__(self, token_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.dense_layers = []  # 存储预创建的Dense层

    def build(self, input_shape):
        """在build阶段预创建所有Dense层"""
        for i in range(len(input_shape)):
            self.dense_layers.append(
                Dense(self.token_dim, activation='linear', name=f'dense_{i}')
            )
        super().build(input_shape)

    def call(self, inputs):
        """重用预创建的Dense层进行特征转换"""
        tokenized = [layer(input) for layer, input in zip(self.dense_layers, inputs)]
        return tf.stack(tokenized, axis=1)
        
        
        
class TokenMixer(Layer):
    def __init__(self,num_T,num_D,num_H,**kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.num_D = num_D
        self.num_H = num_H
        self.d_k = num_D//num_H
    
    def call(self,x):
        x = tf.reshape(x,(-1,self.num_T,self.num_H,self.d_k)) # (B,T,D)->(B,T,H,D/H)
        x = tf.transpose(x,perm=[0,2,1,3])  # (B,H,T,D/H)
        x = tf.reshape(x,(-1,self.num_H,self.num_T*self.d_k)) # (B,H,T*D/H)
        return x 

class PerTokenFFN(Layer):
    def __init__(self, num_T, num_D, expansion_ratio=4, **kwargs):
        super().__init__(**kwargs)
        
        # ReLU Router - Gate权重
        self.gate = Dense(num_T, use_bias=False, name='relu_router_gate')
        
        # 每个expert的FFN
        self.experts = []
        for i in range(num_T):
            self.experts.append([
                Dense(num_D * expansion_ratio, name=f'expert_{i}_fc1'),
                Activation('gelu'),
                Dense(num_D, name=f'expert_{i}_fc2')
            ])
    
    def call(self, x):
        # ReLU Router
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        routing_weights = tf.nn.relu(logits)  # ReLU激活
        routing_mask = tf.cast(routing_weights > 0, tf.float32)  # 路由掩码
        
        # 归一化路由权重
        routing_weights = routing_weights * routing_mask
        routing_weights_sum = tf.reduce_sum(routing_weights, axis=-1, keepdims=True) + 1e-9
        routing_weights = routing_weights / routing_weights_sum
        
        # Expert计算
        expert_outputs = []
        for expert_layers in self.experts:
            h = x
            for layer in expert_layers:
                h = layer(h)
            expert_outputs.append(h)
        
        # Stack: (num_experts, batch, seq_len, num_D)
        expert_outputs = tf.stack(expert_outputs, axis=0)
        
        # 加权组合: (batch, seq_len, 1, num_experts) @ (num_experts, batch, seq_len, num_D)
        routing_weights = tf.expand_dims(routing_weights, axis=-2)  # (batch, seq_len, 1, num_experts)
        expert_outputs = tf.transpose(expert_outputs, [1, 2, 0, 3])  # (batch, seq_len, num_experts, num_D)
        
        output = tf.matmul(routing_weights, expert_outputs)  # (batch, seq_len, 1, num_D)
        output = tf.squeeze(output, axis=-2)  # (batch, seq_len, num_D)
        
        return output
    
class RankMixerLayer(Layer):
    def __init__(self,num_T,num_D,num_H,expansion_ratio,**kwargs):
        super().__init__(**kwargs)
        self.token_mixer = TokenMixer(num_T,num_D,num_H)
        self.per_token_ffn = PerTokenFFN(num_T,num_D,expansion_ratio)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
    def call(self,x):
        mixed_x = self.token_mixer(x)
        x = self.norm1(x+mixed_x)
        x = self.norm2(x+self.per_token_ffn(x))
        return x

class RankMixer(Layer):
    def __init__(self,num_layers,num_T,num_D,num_H,expansion_ratio,token_dim,**kwargs):
        super().__init__(**kwargs)
        self.semantic_tokenization = SemanticTokenization(token_dim)
        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(RankMixerLayer(num_T,num_D,num_H,expansion_ratio))
    def call(self,x):
        x = self.semantic_tokenization(x)
        for layer in self.layers_list:
            x = layer(x)
        return tf.reduce_mean(x,axis=1)
    
    
if __name__ == '__main__':
    inputs = [tf.keras.Input(shape=(8,), name='feature_1'),
        tf.keras.Input(shape=(16,), name='feature_2'),
        tf.keras.Input(shape=(32,), name='feature_3'),
        tf.keras.Input(shape=(32,), name='feature_4')]
    model = RankMixer(num_layers=2,num_T=4,num_D=128,num_H=4,expansion_ratio=4,token_dim=128)
    outputs = model(inputs)
    print(outputs.shape)
