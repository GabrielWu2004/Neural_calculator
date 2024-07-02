import numpy as np
from tqdm import tqdm
from model_architecture import arithmaticTransformer
from data_generation import tokenizer, trainingDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train(model, dataloader, optimizer, criterion, device, report_interval=10, max_iter=1000):
  model.to(device)
  model.train()
  total = len(dataloader)
  with tqdm(enumerate(dataloader), total=total) as pbar:
    for iter, (batch_x, batch_y) in pbar:
      total_loss = 0
      batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L), (B,)
      optimizer.zero_grad()
      out = model.forward(batch_x)[:, -1, :] # (B, vocab_size)
      # print("output shape:", out.shape)
      # print("target shape:", batch_y.shape)
      loss = criterion(out, batch_y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      if (iter+1) % report_interval == 0:
        pbar.set_description(f"loss: {total_loss/report_interval}")
        total_loss = 0
      if (iter+1) == max_iter:
        print("Maximum iteration reached")
        break

if __name__ == "__main__":
  vocab_size, encode, decode = tokenizer("data/training_data_100k.txt")
  training_dataset = trainingDataset("data/training_data_100k.txt", encode)
  training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)

  params = {"vocab_size": vocab_size,
          "context_length": 100,
          "model_size": 8,
          "num_heads": 4,
          "num_blocks": 6,
          "device": device}
  model = arithmaticTransformer(**params)
  learning_rate = 1e-4
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()
  train(model, training_dataloader, optimizer, criterion, device)
