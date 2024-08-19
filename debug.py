""" 
Debug with existing architecture to see if the problem arises from the inference code
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import tokenizer, trainingDataset, get_dataloader
from torch.utils.data import DataLoader

# hyperparameters
batch_size = 512
block_size = 14
vocab_size = 14
eval_interval = 500
learning_rate = 5e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embd = 384
n_head = 8
n_layer = 20
dropout = 0.2
equal_index = 8

torch.manual_seed(1337)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # buffer signifies these are not parameters
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_embd), note: n_embd = n_head * head_size
        out = self.proj(out) # (B, T, n_embd)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Trnasformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

# model
class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd), T words in each batch, each word is a (1, n_embd) vector
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd), [0, ... T-1] --> [pos0 ... posT-1]
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens):
        # input idx: (B, T)
        # output idx: (B, T+1)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond) # equivalent to calling forward()
            # focus only on the last time step 
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probability
            probs = F.softmax(logits, dim=-1) #(B, C)
            # sample from distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1), will end of generating max_new_tokens number of tokens
        return idx



if __name__ == "__main__":
    data_dir = "data/3_digits_addition_padded.txt"
    vocab_size, encode, decode, train_dataloader, val_dataloader = get_dataloader(data_dir, mode="train", batch_size=batch_size)
    model = TransformerModel()
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter, (batch_x, batch_y) in enumerate(train_dataloader):
        total_loss = 0
        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L), (B,L')
        optimizer.zero_grad()
        logits = model.forward(batch_x)[:, equal_index:, :] 
        B, L, C = logits.shape
        logits = logits.reshape(B*L, C)
        targets = batch_y.reshape(B*L)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if iter%100 == 0:
            print(f"iter {iter} Loss: {total_loss/100}")
            total_loss = 0
    print("training complete")

    print("eval")
    model.to(device)
    model.eval()
    total = 0
    num_correct = 0
    for idx, (batch_x, batch_y) in enumerate(val_dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L) and (B, L')
        out = torch.zeros(batch_y.shape).to(device)
        for i in range(batch_x.shape[1]-equal_index):
            input = batch_x[:, :equal_index+i+1]
            logits = model.forward(input)[:, [-1], :] # (B, 1, vocab_size)
            logits = torch.argmax(logits, dim=-1) # (B, 1)
            out[:, i] = logits.squeeze() # (B,)
        matching_output = torch.all(out == batch_y, dim=1)
        num_correct += torch.sum(matching_output).item()
        total += batch_x.shape[0]
    print(f"Total: {num_correct}/{total}")
    print()

# print("teacher forcing eval")
# model.to(device)
# model.eval()
# total = 0
# num_correct = 0
# for idx, (batch_x, batch_y) in enumerate(val_dataloader):
#     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#     logits = model.forward(batch_x)[0] # (B, L', vocab_size)
#     logits = torch.argmax(logits, dim=-1) # (B, L')
#     matching_output = torch.all(logits == batch_y, dim=1)
#     num_correct += torch.sum(matching_output).item()
#     total += batch_x.shape[0]
#     # print(f"Batch {idx}: {torch.sum(matching_output).item()}/{batch_x.shape[0]}")
#     # if idx == 0:
#     #     for item in range(logits.shape[0]):
#     #         print("Model output:", decode(logits[item].tolist()))
#     #         print("Correct answer:", decode(batch_y[item].tolist()))
# print(f"Total: {num_correct}/{total}")
# print()