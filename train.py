import model
import encoder
import torch

class EncodingBlock(torch.nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.5) -> None:
        super().__init__()
        self.encoder = encoder.Encoder(num_layers, model_dim, num_heads, ff_dim, dropout)