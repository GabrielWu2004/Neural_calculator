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


def save_checkpoint_fn(model, optimizer, epoch, loss, checkpoint_path):
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


def train(model, dataloader, optimizer, scheduler, criterion, device, model_name, report_interval=100, max_iter=1e6, save_checkpoint=False, checkpoint_path=None, checkpoint_interval=None, resume_checkpoint=False, checkpoint_dir='model'):
  # Make checkpoint directory
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  # Resume from checkpoint
  start_iter = 0
  if resume_checkpoint:
    model, optimizer, start_iter, _ = load_checkpoint(checkpoint_path, model, optimizer)
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device)
    print(f"Resuming training from iteration {start_iter}")
  else:
    print("Training model from scratch")
  
  # Training loop
  model.to(device)
  model.train()
  total = len(dataloader)
  # num_streams = 16
  # streams = [torch.cuda.Stream() for _ in range(num_streams)] # Create CUDA streams

  print("Training begin")
  with tqdm(enumerate(dataloader, start=start_iter), total=total) as pbar:
    for iter, (batch_x, batch_y) in pbar:
      total_loss = 0
      batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L), (B,)
      
      # Distribute batches across streams
      # stream_idx = iter % num_streams
      # with torch.cuda.stream(streams[stream_idx]):
      optimizer.zero_grad()
      # print("input:", batch_x)
      # print("target:", batch_y)
      out = model.forward(batch_x)[:, -1, :] # (B, vocab_size)
      # print("out:", out, out.shape)
      loss = criterion(out, batch_y)
      # print("loss:", loss)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      scheduler.step()
      # if (iter + 1) % num_streams == 0:
      #   torch.cuda.synchronize()

      # Reporting loss
      if (iter+1) % report_interval == 0:
        pbar.set_description(f"loss: {total_loss/report_interval}")
        total_loss = 0
      if save_checkpoint:
        if (iter+1) % checkpoint_interval == 0:
          checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint_iter_{iter+1}.pth')
          save_checkpoint_fn(model, optimizer, iter+1, loss.item(), checkpoint_path)
      if (iter+1) == max_iter:
        print("Maximum iteration reached")
        break

  final_model_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
  torch.save(model, final_model_path)
  print(f"Final model saved to {final_model_path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
  # Load data
  max_length = 10
  vocab_size, _, _, dataloader = get_dataloader("data/toy_data_10k.txt", mode="train_single", batch_size=1, max_length=max_length)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Build model
  params = {"vocab_size": vocab_size,
          "context_length": max_length,
          "model_size": 16,
          "num_heads": 8,
          "num_blocks": 6,
          "device": device}
  model = arithmaticTransformer(**params)
  print(f'The model has {count_parameters(model):,} trainable parameters')
  
  # Train model
  learning_rate = 5e-4
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.05)
  criterion = nn.CrossEntropyLoss()
  train(model, dataloader, optimizer, scheduler, criterion, device, max_iter=5e3, model_name="testing_model")


if __name__ == "__main__":
  main()
