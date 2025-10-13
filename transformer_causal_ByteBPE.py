import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import text
from tokenizers import ByteLevelBPETokenizer


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 128  #context
BATCH = 8
EMBD = 256 #embeding size
N_LAYERS = 8
N_HEAD = 16
LR = 3e-4
EPOCHS = 1000 #epochs
PRINT_EVERY = EPOCHS//10

# dataset, toy one for testing (few sentences)
text = text

# tokenizer Byte level BPE
tokenizer = ByteLevelBPETokenizer(
    "tokenizer_bpe/vocab.json",
    "tokenizer_bpe/merges.txt"
)

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size()

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.2 * len(data)) #20% val
train_data = data[n:]
val_data = data[:n]

def get_batch(split="train"):
    source = train_data if split=="train" else val_data
    ix = torch.randint(0, len(source) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([source[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([source[i+1:i+SEQ_LEN+1] for i in ix])
    return x.to(device), y.to(device)

# tiny toy transofrmer (causal, GPT like)
class TinyBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        

        # projections
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)
        self.wo = nn.Linear(n_embd, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

        # feed forward
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x, mask, train=True):
        B, T, C = x.size()

        # self-attention
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).permute(0,2,1,3)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).permute(0,2,1,3)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).permute(0,2,1,3)

        att = (q@k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att@v
        out = out.permute(0,2,1,3).contiguous().view(B, T, C)
        out = self.wo(out)
        if train == True: #only apply dropout to train
            out = self.dropout(out)
        x = x + out
        x = self.ln1(x)

        # feed forward
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ln2(x)
        return(x)

class TinyLM(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embd=EMBD, n_layers=N_LAYERS, n_head=N_HEAD):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.layers = nn.ModuleList([TinyBlock(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, idx, targets=None, train=True):
        B, T = idx.size()
        assert T <= self.seq_len, "sequence too long for model" #change the seq_len or window
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # causal mask 1 = keep, 0 = mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask, train=train)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None: #if we dont have targets (eval, gen mode)
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return(logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            B, T = idx.size()
            if T > self.seq_len:
                idx_cond = idx[:, -self.seq_len:]
            else:
                idx_cond = idx
            logits = self.forward(idx_cond, train=False)
            logits = logits[:, -1, :]/max(1e-9, temperature)
            # top k filter 
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter(1, ix, v)
                probs = F.softmax(probs, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return(idx)

#train
model = TinyLM(vocab_size=vocab_size, seq_len=SEQ_LEN).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

best_val = float("inf")
patience = 20
no_improve = 0

print(f"Vocab size: {vocab_size}, device: {device}")
for epoch in range(1, EPOCHS+1):
    model.train()
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % PRINT_EVERY == 0 or epoch==1:
        # compute val loss
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch("val")
            _, vloss = model(vx, vy)
        print(f"epoch {epoch} train_loss {loss.item():.4f} val_loss {vloss.item():.4f}")
        start = "Bonjour"
        idx = torch.tensor([encode(start)], dtype=torch.long).to(device)
        gen = model.generate(idx, max_new_tokens=10)[0].tolist()
        print(" ->", decode(gen))
        # checkpointing & early stopping
        if vloss.item() < best_val:
            best_val = vloss.item()
            torch.save(model.state_dict(), "best_tinylm.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping (no improvement)")
                break
    scheduler.step()

# save
torch.save(model.state_dict(), "tiny_lm.pt")
print("Saved model tiny_lm.pt")
