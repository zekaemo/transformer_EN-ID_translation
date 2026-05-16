class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i       = torch.arange(0, self.d_model, 2).float() #list of number 0 to dmodel, 2 steps
        denominator  = torch.pow(10000, even_i / self.d_model) #10000^(i/dmodel)
        position     = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1) #change dimension from (max_seq_len, ) to (max_seq_len, 1)
        even_PE      = torch.sin(position / denominator) # eg.[[1,3], [5,7]]
        odd_PE       = torch.cos(position / denominator) # eg.[[2,4], [6,8]]
        stacked      = torch.stack([even_PE, odd_PE], dim=2) # join both PE alternately [[[1,2], [3,4]],[[5,6],[7,8]]]
        PE           = torch.flatten(stacked, start_dim=1, end_dim=2) # eg.[[1,2,3,4], [5,6,7,8]]
        return PE   # (max_seq_len, d_model)
    
    
class SentenceEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_sequence_length, pad_id, dropout=0.1):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout         = nn.Dropout(p=dropout)
    def forward(self, token_ids):  # token_ids: (B, T) — already integers from DataLoader
        x   = self.embedding(token_ids)                          # (B, T, d_model)
        pos = self.position_encoder().to(token_ids.device)       # (MAX_SEQ_LEN, d_model)
        pos = pos[:token_ids.size(1), :]                         # slice to actual T
        return self.dropout(x + pos)