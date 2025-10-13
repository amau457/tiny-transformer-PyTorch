import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import text


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 128  #context
BATCH = 32
EMBD = 256 #embeding size
N_LAYERS = 8
N_HEAD = 4
LR = 3e-4
EPOCHS = 1000 #epochs
PRINT_EVERY = EPOCHS//10

# dataset, toy one for testing (few sentences)
text = text* 200  # augmentation (overfit but we have to with this size of ds)

# tokenizer (character level)
chars = sorted(list(set(text))) #we separate character one by one
stoi = {ch:i for i,ch in enumerate(chars)} #char to ids vocab
itos = {i:ch for ch,i in stoi.items()} #ids to char vocab
vocab_size = len(chars)

def encode(s):
    #encode with tokenizer
    return([stoi[c] for c in s])

def decode(ids):
    #decode with tokenizer
    return("".join(itos[i] for i in ids))

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size=BATCH, seq_len=SEQ_LEN):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return(x.to(device), y.to(device))

# tiny toy transofrmer (causal, GPT like)
class TinyBlock(nn.Module):
    def __init__(self, n_embd, n_head):
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

        # feed forward
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x, mask):
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.seq_len, "sequence too long for model" #change the seq_len or window
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # causal mask 1 = keep, 0 = mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None: #if we dont have targets (eval, gen mode)
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return(logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            B, T = idx.size()
            if T > self.seq_len:
                idx_cond = idx[:, -self.seq_len:]
            else:
                idx_cond = idx
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)# sample
            idx = torch.cat([idx, next_token], dim=1)
        return(idx)

#train
model = TinyLM(vocab_size=vocab_size, seq_len=SEQ_LEN).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"Vocab size: {vocab_size}, device: {device}")
for step in range(1, EPOCHS+1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % PRINT_EVERY == 0 or step == 1:
        print(f"step {step:4d} loss {loss.item():.4f}")
        # test
        start = "Bonjour"
        idx = torch.tensor([encode(start)], dtype=torch.long).to(device)
        gen = model.generate(idx, max_new_tokens=80)[0].tolist()
        print(" ->", decode(gen))

# save
torch.save(model.state_dict(), "tiny_lm.pt")
print("Saved model tiny_lm.pt")
