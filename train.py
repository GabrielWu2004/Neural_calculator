import numpy as np
import os
from tqdm import tqdm
from model_architecture import arithmaticTransformer
from data import tokenizer, trainingDataset, get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


equal_index = 8 
torch.manual_seed(1337)

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


def train(model, dataloader, optimizer, scheduler, device, model_name, report_interval=100, max_iter=1e6, save_checkpoint=False, checkpoint_path=None, checkpoint_interval=None, resume_checkpoint=False, checkpoint_dir='model'):
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
      batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L), (B,L')
      # print("batch X shape:", batch_x.shape)
      # print("batch Y shape:", batch_y.shape)
      optimizer.zero_grad()
      logits = model.forward(batch_x)[:, equal_index:, :] 
      B, L, C = logits.shape
      # print("output shape before reshape:", logits.shape) # (B, L', vocab_size)
      logits = logits.reshape(B*L, C)
      targets = batch_y.reshape(B*L)
      # print("output shape:", logits.shape)
      # print("target shape:", targets.shape)
      loss = F.cross_entropy(logits, targets)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      scheduler.step()

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


def val(model, dataloader, device, decode):
  model.to(device)
  model.eval()
  total = 0
  num_correct = 0
  for idx, (batch_x, batch_y) in enumerate(dataloader):
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    logits = model.forward(batch_x)[:, equal_index:, :] # (B, L', vocab_size)
    logits = torch.argmax(logits, dim=-1) # (B, L')
    matching_output = torch.all(logits == batch_y, dim=1)
    num_correct += torch.sum(matching_output).item()
    total += batch_x.shape[0]
    print(f"Batch {idx}: {torch.sum(matching_output).item()}/{batch_x.shape[0]}")
    if idx == 0:
      for item in range(logits.shape[0]):
        print("Model output:", decode(logits[item].tolist()))
        print("Correct answer:", decode(batch_y[item].tolist()))

  print(f"Total: {num_correct}/{total}")

def val_iter(model, dataloader, device, decode):
  model.to(device)
  model.eval()
  total = 0
  num_correct = 0
  for idx, (batch_x, batch_y) in enumerate(dataloader):
    batch_x, batch_y = batch_x.to(device), batch_y.to(device) # (B, L) and (B, L')
    out = torch.zeros(batch_y.shape).to(device)
    # print("expected output shape:", out.shape)
    for i in range(batch_x.shape[1]-equal_index):
      input = batch_x[:, :equal_index+i+1]
      # print("input shape:", input.shape)
      print("actual input:", decode(input[0].tolist()))
      logits = model.forward(input)[:, [-1], :] # (B, 1, vocab_size)
      # print("logit shape:", logits.shape)
      logits = torch.argmax(logits, dim=-1) # (B, 1)
      # print("logit shape after selection:", logits.shape)
      print("model output:", decode(logits[0].tolist()))
      out[:, i] = logits.squeeze() # (B,)
    matching_output = torch.all(out == batch_y, dim=1)
    num_correct += torch.sum(matching_output).item()
    total += batch_x.shape[0]
    print(f"Batch {idx}: {torch.sum(matching_output).item()}/{batch_x.shape[0]}")
    if idx == 0:
      for item in range(out.shape[0]):
        print("Model output:", decode(out[item].tolist()))
        print("Correct answer:", decode(batch_y[item].tolist()))
    break
  print(f"Total: {num_correct}/{total}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
  data_dir = "data/3_digits_addition_padded.txt"
  vocab_size, encode, decode, train_dataloader, val_dataloader = get_dataloader(data_dir, mode="train", batch_size=256)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  # Build model
  params = {"vocab_size": vocab_size,
          "context_length": 14,
          "model_size": 64,
          "num_heads": 8,
          "num_blocks": 6,
          "device": device}
  model = arithmaticTransformer(**params)
  print(f'The model has {count_parameters(model):,} trainable parameters')
  
  # Train model
  learning_rate = 5e-4
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.05)
  train(model, train_dataloader, optimizer, scheduler, device, max_iter=1e6, model_name="testing_model")
  val_iter(model, val_dataloader, device, decode)


if __name__ == "__main__":
  main()
