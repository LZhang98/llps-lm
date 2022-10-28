import torch
import math

class Encoder (torch.nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.5) -> None:
        super().__init__()
        self.model_type = 'Encoder'
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)

        self.pool_flatten = torch.nn.Sequential(
                                torch.nn.AvgPool1d(kernel_size=10, stride=10),
                                torch.nn.Flatten()
        )
        #move fc layers out in the future
        # self.fc1 = torch.nn.Linear(1280, 240)
        # self.fc2 = torch.nn.Linear(240, 48)
        # self.final = torch.nn.Linear(48, 1)
        self.dense = torch.nn.Sequential(
                        torch.nn.Linear(1336 * (model_dim // 10), 1280),
                        torch.nn.ReLU(),
                        torch.nn.Linear(1280, 240),
                        torch.nn.ReLU(),
                        torch.nn.Linear(240, 48),
                        torch.nn.ReLU(),
                        torch.nn.Linear(48, 1),
                        torch.nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.pool_flatten(x)
        x = self.dense(x)
        return x


class PositionalEncoding (torch.nn.Module):
    def __init__(self, model_dim, dropout=0.5, max_len=5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == '__main__':

    num_layers = 1
    model_dim = 12
    num_heads = 3
    ff_dim = 12

    print('test')
    encoder = Encoder(num_layers, model_dim, num_heads, ff_dim)
    pos_encoder = PositionalEncoding(model_dim)
    x = torch.randn((10, 1))
    pe_x = pos_encoder(x)
    print(pe_x.size())   
    x = encoder(x) 
    print(encoder(x))
    print(x.size())