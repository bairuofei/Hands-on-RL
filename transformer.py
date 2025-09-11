import torch


import torch.nn as nn



# 1. embedding
# 2. position encoding
# 3. transformer layer
#     3.1 Q, K, V linear projection
#     3.2 Scaled Dot-Product Attention

    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, latent_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        self.qkv = nn.Linear(embed_dim, latent_dim * 3)
        self.scale = latent_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):  # x is a tensor of shape (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x) # (batch_size, seq_len, latent_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.latent_dim)
        
        scaled_weights = self.scale * torch.matmul(qkv[:, :, 0, :], qkv[:, :, 1, :].transpose(-1, -2))  # (batch_size, seq_len, seq_len)
        unified_weights = self.softmax(scaled_weights)
        return torch.matmul(unified_weights, qkv[:, :, 2, :])  # batch_size, seq_len, latent_dim
    
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.v_dim = latent_dim // num_heads
        self.num_heads = num_heads
        
        
        self.QKV = nn.Linear(latent_dim, 3*latent_dim) # q, k, v
        self.scale = self.v_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x):
        qkv = self.QKV(x)  # (batch_size, seq_len, latent_dim * 3 * num_heads)
        batch_size, seq_len, _ = qkv.shape
        # qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3, self.latent_dim)
        qkv = qkv.reshape(batch_size, self.num_heads, seq_len, 3, self.v_dim)
        
        scaled_weights = torch.matmul(qkv[:, :, :, 0, :], qkv[:, :, :, 1, :].transpose(-1, -2)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        unified_weights = self.softmax(scaled_weights)
        out = torch.matmul(unified_weights, qkv[:, :, :, 2, :])  # batch_size, num_heads, seq_len, v_dim
        out = out.permute(0, 2, 1, 3)
        return self.out(out.reshape(batch_size, seq_len, self.latent_dim))  # (batch_size, seq_len, embed_dim)


class Transformer(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super().__init__()
        self.multihead_atten = MultiHeadAttention(latent_dim, num_heads)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, 4*latent_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),  # 提升训练稳定度
            nn.Linear(4*latent_dim, latent_dim, bias=True)
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        atten_out = self.multihead_atten(x)
        x = self.norm1(x + atten_out)
        ffn_out = self.ffn(x)
        return self.norm2(x+ffn_out)
        

        
if __name__ == "__main__":
    # test dimentions
    x = torch.randn(100, 10, 1024)
    
    model = Transformer(latent_dim=1024, num_heads=8)
    
    out = model(x)
    print(out.shape)  # (100, 10, 1024)