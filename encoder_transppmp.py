import torch
import numpy as np

# TODO: determine if training flags are necessary
# TODO: implement torch.permute

class Encoder (torch.nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, rate=0.5) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.pos_encoding = positional_encoding(1000, self.model_dim)
        self.encoding_layers = [EncoderNN(model_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(rate=rate)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.encoding_layers[i](x, mask)
        return x

def get_angles(pos, i, model_dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(model_dim))
    return pos * angle_rates

def positional_encoding(position, model_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(model_dim)[np.newaxis, :], model_dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding.type(torch.FloatTensor)

class EncoderNN (torch.nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, rate=0.5) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ffn = feed_forward_network(model_dim, ff_dim)
        self.layernorm1 = torch.nn.LayerNorm(eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(eps=1e-6)
        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):

        # multihead attention + dropout
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output)

        # combine and layernorm
        out1 = self.layernorm1(x + attention_output)

        # feed forward network + dropout
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        # combine and layernorm
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

def feed_forward_network (model_dim, ff_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(ff_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(model_dim)
    )

class MultiHeadAttention (torch.nn.Module):

    def __init__(self, model_dim, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim

        assert model_dim % self.num_heads == 0
        self.depth = self.model_dim // self.num_heads
        self.wq = torch.nn.Linear(self.model_dim)
        self.wk = torch.nn.Linear(self.model_dim)
        self.wv = torch.nn.Linear(self.model_dim)
        self.FC = torch.nn.Linear(self.model_dim)
    
    def split_heads (self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return torch.permute(x, (0, 2, 1, 3))
    
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(dim=0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = torch.permute(scaled_attention, (0, 2, 1, 3))
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        output = self.FC(concat_attention)
        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask=None):
    qk = torch.matmul(q, k)
    k_dim = k.size(dim=-1)
    scaled_attention_logits = qk / torch.sqrt(k_dim)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = torch.softmax(scaled_attention_logits, axis=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

if __name__ == '__main__':
    ...