def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # (B, H, T_q, T_k)
    if mask is not None:
        # mask shape (B, 1, ?, T_k) broadcasts over H automatically
        scaled = scaled + mask
    attention = F.softmax(scaled, dim=-1)   # (B, H, T_q, T_k)
    values = torch.matmul(attention, v)     # (B, H, T_q, d_k)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model   = d_model #size of embedding vectors
        self.num_heads = num_heads #number of heads
        self.head_dim  = d_model // num_heads #vector size in attention head (32)
        self.qkv_layer    = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model) # Z (output of z*w_o)

    def forward(self, x, mask=None, return_attention=False):
        B, T, _ = x.size() #B= how much sentence in one batch (batch size), T= how much token in one sentence (sequence length)
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3) #split into heads
        q, k, v = qkv.chunk(3, dim=-1) #separate qkv
        values, attention = scaled_dot_product(q, k, v, mask) #compute attention
        values = values.permute(0, 2, 1, 3).reshape(B, T, self.d_model) #combine all heads
        out = self.linear_layer(values) #times W_O
        if return_attention:
            return out, attention
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.kv_layer     = nn.Linear(d_model, 2 * d_model)
        self.q_layer      = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None, return_attention=False):
        B, T_src, _ = x.size()
        T_tgt = y.size(1) #target sequence length
        kv = self.kv_layer(x).reshape(B, T_src, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3) #compute KV from encoder, then split
        q  = self.q_layer(y).reshape(B, T_tgt, self.num_heads, self.head_dim).permute(0, 2, 1, 3) #compute Q from decoder, then split
        k, v = kv.chunk(2, dim=-1) #separate kv
        values, attention = scaled_dot_product(q, k, v, mask) #compute attention
        values = values.permute(0, 2, 1, 3).reshape(B, T_tgt, self.d_model) #combine all heads
        out = self.linear_layer(values) #times W_O
        if return_attention:
            return out, attention
        return out