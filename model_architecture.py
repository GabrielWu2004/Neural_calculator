import torch
import torch.nn as nn
import torch.nn.functional as F

class attentionHead(nn.Module):
  def __init__(self, model_size, head_size, context_length):
    super(attentionHead, self).__init__()
    self.key = nn.Linear(model_size, head_size, bias=False)
    self.query = nn.Linear(model_size, head_size, bias=False)
    self.value = nn.Linear(model_size, head_size, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))

  def forward(self, x):
    """
    x: (B, L, N)
    return: (B, L, H)
    """
    B, L, N = x.shape
    key = self.key(x) # (B, L, H)
    query = self.query(x) # (B, L, H)
    value = self.value(x) # (B, L, H)
    attention = torch.matmul(query, key.transpose(-2, -1))*N**(-0.5) # (B, L, L)
    attention.masked_fill(self.tril[:L, :L] == 0, float('-inf'))
    attention = F.softmax(attention, dim=-1)
    return torch.matmul(attention, value) # (B, L, H)

class multiHeadAttention(nn.Module):
  def __init__(self, model_size, num_heads, context_length):
    super(multiHeadAttention, self).__init__()
    if model_size % num_heads != 0:
      raise ValueError("model_size must be divisible by head_size")
    head_size = model_size // num_heads
    self.heads = nn.ModuleList([attentionHead(model_size, head_size, context_length) for _ in range(num_heads)])
    self.linear = nn.Linear(model_size, model_size)

  def forward(self, x):
    """
    x: (B, L, N)
    return: (B, L, N)
    """
    return self.linear(torch.cat([h(x) for h in self.heads], dim=-1))

class feedForward(nn.Module):
  def __init__(self, model_size):
    super(feedForward, self).__init__()
    self.linear1 = nn.Linear(model_size, 4*model_size)
    self.linear2 = nn.Linear(4*model_size, model_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    """
    x: (B, L, N)
    return: (B, L, N)
    """
    return self.linear2(self.relu(self.linear1(x)))

class attentionBlock(nn.Module):
  def __init__(self, model_size, num_heads, context_length):
    super(attentionBlock, self).__init__()
    self.multiHeadAttention = multiHeadAttention(model_size, num_heads, context_length)
    self.feedForward = feedForward(model_size)
    self.norm1 = nn.LayerNorm(model_size)
    self.norm2 = nn.LayerNorm(model_size)

  def forward(self, x):
    """
    x: (B, L, H)
    return: (B, L, H)
    """
    x = self.norm1(x + self.multiHeadAttention(x))
    x = self.norm2(x + self.feedForward(x))
    return x

class arithmaticTransformer(nn.Module):
  def __init__(self, vocab_size, context_length, model_size, num_heads, num_blocks, device):
    super(arithmaticTransformer, self).__init__()
    self.context_length = context_length
    self.device = device
    self.embedding = nn.Embedding(vocab_size, model_size)
    self.positionalEmbedding = nn.Embedding(context_length, model_size)
    self.attentionBlocks = nn.Sequential(*[attentionBlock(model_size, num_heads, context_length) for _ in range(num_blocks)])
    self.linear = nn.Linear(model_size, vocab_size)
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def to(self, device):
    self.device = device
    return super().to(device)

  def forward(self, x):
    """
    x: (B, L)
    return: (B, L, vocab_size)
    """
    B, L = x.shape
    token_embd = self.embedding(x) # (B, L, model_size)
    pos_embd = self.positionalEmbedding(torch.arange(L, device=self.device)) # (L, model_size)
    x = token_embd + pos_embd # (B, L, model_size)
    x = self.attentionBlocks(x) # (B, L, model_size)
    return self.linear(x) # (B, L, vocab_size)
    # Note: each token in x is predicting what the next token is

  @torch.no_grad()
  def generate(self, x, encode):
    """
    x: (1, L)
    return: (1, L') where L' is the length of the generated sequence
    """
    self.eval()
    B, L = x.shape
    for _ in range(self.context_length - L):
      # print("input:", x)
      logits = self.forward(x) # (1, L', vocab_size)
      last_logit = logits[:, -1, :] # (1, vocab_size)
      # print("last logits:", last_logit)
      out_token = torch.argmax(last_logit, dim=-1) # (1,)
      # print("output:", out_token)
      x = torch.cat([x, out_token.unsqueeze(1)], dim=1) # (1, L+1)
      # print("new sequence:", x)
      if out_token.item() == encode("$")[0]:
        break
      # print()
    # print("return sequence:", x[:, L:])
    return x[:, L:]
