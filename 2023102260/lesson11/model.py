import math
import torch
import torch.nn as nn

# ===================== 位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ===================== Skeleton Transformer 模型 =====================
class SkeletonTransformer(nn.Module):
    def __init__(
        self,
        input_dim=132,
        target_frames=30,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=6,
        dropout=0.1
    ):
        super().__init__()
        self.target_frames = target_frames
        self.linear_emb = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=target_frames)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.linear_emb(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits