import numpy as np
import os
from tqdm import tqdm
from model_architecture import arithmaticTransformer
from data_generation import tokenizer, trainingDataset, get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
  checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
  }
  torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  return model, optimizer, epoch, loss


def train(model, dataloader, optimizer, scheduler, criterion, device, report_interval=10, max_iter=1e6, save_checkpoint=False, checkpoint_path=None, checkpoint_interval=None, resume_checkpoint=False, checkpoint_dir='model'):
  # Make checkpoint directory
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  # Resume from checkpoint
  start_iter = 0
  if resume_checkpoint:
    model, optimizer, start_iter, _ = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"Resuming training from iteration {start_iter}")
  else:
    print("Training model from scratch")
  # Training loop
  model.to(device)
  model.train()
  total = len(dataloader)

  print("Training begin")
  with tqdm(enumerate(dataloader, start=start_iter), total=total) as pbar:
    for iter, (batch_x, batch_y) in pbar:
      total_loss = 0
      batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L), (B,)
      optimizer.zero_grad()
      out = model.forward(batch_x)[:, -1, :] # (B, vocab_size)
      loss = criterion(out, batch_y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      scheduler.step()

      if (iter+1) % report_interval == 0:
        pbar.set_description(f"loss: {total_loss/report_interval}")
        total_loss = 0
      if save_checkpoint:
        if (iter+1) % checkpoint_interval == 0:
          checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iter+1}.pth')
          save_checkpoint(model, optimizer, iter+1, loss.item(), checkpoint_path)
      if (iter+1) == max_iter:
        print("Maximum iteration reached")
        break

  final_model_path = os.path.join(checkpoint_dir, f'final_model_{total}.pth')
  torch.save(model, final_model_path)
  print(f"Final model saved to {final_model_path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
  # Load data
  max_length = 10
  vocab_size, _, _, dataloader = get_dataloader("data/training_data_1M.txt", mode="train_padded", batch_size=64, max_length=max_length, shuffle=False)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Build model
  params = {"vocab_size": vocab_size,
          "context_length": max_length,
          "model_size": 32,
          "num_heads": 4,
          "num_blocks": 8,
          "device": device}
  model = arithmaticTransformer(**params)
  print(f'The model has {count_parameters(model):,} trainable parameters')
  
  # Train model
  learning_rate = 5e-4
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)
  criterion = nn.CrossEntropyLoss()
  train(model, dataloader, optimizer, scheduler, criterion, device)


if __name__ == "__main__":
  main()
