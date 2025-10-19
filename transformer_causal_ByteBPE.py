import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import text
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 128  #context
BATCH = 16
EMBD = 512 #embeding size
N_LAYERS = 6
N_HEAD = 8
LR = 3e-4 #base lr (before cosine and after linear)
EPOCHS = 40
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
total_tokens = data.size(0)
print("nb of tokens: ", total_tokens)


torch.manual_seed(42)
num_chunks = total_tokens // SEQ_LEN  #nb of sequences (chunks) of len seq_len in data
trimmed = data[: num_chunks * SEQ_LEN]
chunks = trimmed.view(num_chunks, SEQ_LEN)
perm = torch.randperm(num_chunks) 
chunks = chunks[perm]  #shuffle

n_val_chunks = int(0.2 * num_chunks) # 20% of val
val_chunks = chunks[:n_val_chunks]
train_chunks = chunks[n_val_chunks:]
val_data = val_chunks.contiguous().view(-1)
train_data = train_chunks.contiguous().view(-1)



def get_batch(split="train"):
    source = train_data if split=="train" else val_data
    ix = torch.randint(0, len(source) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([source[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([source[i+1:i+SEQ_LEN+1] for i in ix])
    return x.to(device), y.to(device)

# tiny toy transofrmer (causal, GPT like)
class TinyBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout = 0.2):
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
        if train == True: #only apply dropout to train
            ff_out = self.dropout(ff_out)
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
        if not hasattr(self, "_causal_mask") or self._causal_mask.size(2) < T:
            full = torch.tril(torch.ones(self.seq_len, self.seq_len, device=idx.device))
            self._causal_mask = full.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)
        mask = self._causal_mask[:, :, :T, :T]
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
    
def batchify(data, batch_size, device):
    n = data.size(0) // batch_size * batch_size
    data = data[:n]
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch_from_stream(source, i, seq_len):
    seq_len = min(seq_len, source.size(0) - 1 - i)
    x = source[i:i+seq_len]
    y = source[i+1:i+1+seq_len]
    return x.t().contiguous(), y.t().contiguous()

def get_lr_at(global_step):
    if global_step < warmup_steps:
        return base_lr * (global_step / max(1, warmup_steps))
    else:
        progress = (global_step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * progress))

#train
train_stream = batchify(train_data, BATCH, device)
steps_per_epoch = (train_stream.size(0) - 1) // SEQ_LEN
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(300, int(0.03 * total_steps))
base_lr = LR
eta_min = 1e-6


model = TinyLM(vocab_size=vocab_size, seq_len=SEQ_LEN).to(device)
model.head.weight = model.tok_emb.weight   #tie embedding et heads 
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.001)

val_stream = batchify(val_data, BATCH, device)
val_steps = (val_stream.size(0) - 1) // SEQ_LEN
best_val = float("inf")
patience = 5
no_improve_checks = 0
global_step = 0
lr_list = []  #for ploting
step_list = [] #for ploting
val_loss_list = [] #for ploting
train_loss_list = [] #for ploting
epoch_list = [] #for ploting
train_loss_list_step = [] #for ploting


print(f"Vocab size: {vocab_size}, device: {device}")
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    offset = random.randint(0, SEQ_LEN - 1)
    max_steps = (train_stream.size(0) - 1 - offset) // SEQ_LEN
    with tqdm(range(max_steps), desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") as pbar:
        for step in pbar:
            i = offset + step * SEQ_LEN
            xb, yb = get_batch_from_stream(train_stream, i, SEQ_LEN)  # already on device
            lr = get_lr_at(global_step)
            lr_list.append(lr)
            step_list.append(global_step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            #gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}",
                               'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
            train_loss_list_step.append(loss.item())
        avg_train_loss = epoch_loss / max_steps if max_steps > 0 else float("nan")
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for step in range(val_steps):
                i = step * SEQ_LEN
                vx, vy = get_batch_from_stream(val_stream, i, SEQ_LEN)
                _, vloss = model(vx, vy, train=False)   # IMPORTANT: train=False
                val_loss_sum += vloss.item()
        vloss_mean = val_loss_sum / val_steps
        #print(f"epoch {epoch} train_loss {avg_train_loss:.4f} val_loss {vloss_mean:.4f}")
        val_loss_list.append(vloss_mean)
        train_loss_list.append(avg_train_loss)
        epoch_list.append(epoch)
        if vloss_mean < best_val - 1e-4:
            best_val = vloss_mean
            torch.save(model.state_dict(), "best_tinylm.pt")
            no_improve_checks = 0
        else:
            no_improve_checks += 1
            if no_improve_checks >= patience:
                print("Early stopping (no improvement over checks).")
                break

    if epoch % PRINT_EVERY == 0 or epoch==1:
        print(f"epoch {epoch} train_loss {avg_train_loss:.4f} val_loss {vloss_mean:.4f}")
        start = "je"
        idx = torch.tensor([encode(start)], dtype=torch.long).to(device)
        gen = model.generate(idx, max_new_tokens=10)[0].tolist()
        print(" ->", decode(gen))

plt.figure(1) #learning rate graph
plt.plot(step_list, lr_list)
plt.title('lr over steps')
plt.show(block = False)

plt.figure(2) #loss graph on steps
plt.plot(step_list, train_loss_list_step, label='train loss')
plt.legend()
plt.title('train loss over steps')
plt.show(block=False)

plt.figure(3) #loss graph
plt.plot(epoch_list, train_loss_list, label='train loss')
plt.plot(epoch_list, val_loss_list, label='val loss')
plt.legend()
plt.title('loss over epochs')
plt.show()


# save
torch.save(model.state_dict(), "tiny_lm.pt")
print("Saved model tiny_lm.pt")
