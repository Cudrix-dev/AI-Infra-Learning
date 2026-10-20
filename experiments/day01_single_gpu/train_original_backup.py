import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# B: Batch size; T: 序列长度; V: 词表大小; D: Embedding维度
B, T, V, D = 16, 64, 128, 128
WARMUP, STEPS = 10, 100
assert torch.cuda.is_available(), "this experiment needs CUDA GPU."
device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.manual_seed(42)


class TinyTransformer(nn.Moduel):
    """
    Model architecture:
    input: token id & position id
    ↓
    embedding layer(token_embedding & position embedding) [B L D]
    ↓
    Two Transformer Encoder layers
    ↓
    layer norm(used to stable gradient because layer above uses residual connection)
    ↓
    Linear layer (header) projecting to wordtable space [B L V]
    ↓
    Softmax
    """

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(T, D)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=D,  # the dimension of embedded
                    nhead=4,  # multi head attention
                    dim_feedforward=4 * D,
                    dropout=0.0,
                    batch_first=True,  # [B,L,D]
                    norm_first=True,  # layer_norm between attention layer and forward layer
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, tokens):
        length = tokens.size(1)  # [B T]
        pos = torch.arrange(length, device=tokens.device)
        # [B T D] + [T D], boardcast across  batch dimension
        h = self.token_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, src_mask=self.causal_mask[:length, :length])
        return self.head(
            self.norm(h)
        )  # [B: batch T: length V: probability of each word]


def build():
    model = TinyTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False, fused=False
    )
    return model, optimizer


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))  # [B*T,V] & [B*T,]
    loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    print("PyTorch: ", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    model, optimizer = build()
    # torch.arrange generates 1 dim B lenght vector, [:, None] changes it to 2 dim [B,1] column vector
    starts = torch.arrange(B, device=device)[:, None]
    # utilize boardcast machanisium
    # e.g. seq=[[0,1,2,3],
    #           [1,2,3,4]]
    seq = (starts + torch.arrange(T + 1, device=device)) % V
    # input x: without last element
    x = seq[:, :-1].contiguous()
    # label y: without first element
    y = seq[:, 1:].contiguous()
    # contiguous() is used to make a sliced tensor contiguous in memory
    for _ in range(WARMUP):
        train_step(model, optimizer, x, y)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "day01_single_gpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
