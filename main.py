import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self,vocab_size, n_segments,max_len, embed_dim, dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim) # token embed, segment embed, pos embed
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self,seq,seg):
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp) # embed added
        return embed_val
    

class BERT(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, n_layer, attn_heads, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=attn_heads, dim_feedforward=embed_dim * 4)   # setting up the Transformer encoder part of your BERT model
        self.encoder_block = nn.TransformerEncoder( encoder_layer, num_layers=n_layer)


    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block
        return out




if __name__ == "__main__": # VALUES ALL FROM THE PAPER
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1


sample_seq =  torch.randint(high = VOCAB_SIZE, size =[MAX_LEN,])
sample_seg = torch.randint(high = N_SEGMENTS, size=[MAX_LEN, ])
print(sample_seq)
print(sample_seg)

embedding = BERTEmbedding (VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
embedding_tensor = embedding(sample_seq, sample_seg)
print(embedding_tensor.size())

bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)

out = bert(sample_seq, sample_seg)

print(out)
