import numpy as np
import os
from tqdm import tqdm
from model_architecture import arithmaticTransformer
from debug import TransformerModel
from data import tokenizer, trainingDataset, get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


equal_index = 8
context_length = 14 
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
  
  print("Training begin")
  with tqdm(enumerate(dataloader, start=start_iter), total=total) as pbar:
    for iter, (batch_x, batch_y) in pbar:
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


def val_tf(model, dataloader, device, decode):
  """
  Evaluation: teacher-forcing style
  """
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
    # print(f"Batch {idx}: {torch.sum(matching_output).item()}/{batch_x.shape[0]}")
    # Optionally print out model output versus correct answer
    # if idx == 0:
    #   for item in range(logits.shape[0]):
    #     print("Model output:", decode(logits[item].tolist()))
    #     print("Correct answer:", decode(batch_y[item].tolist()))
  print(f"Teacher forcing inference result: {num_correct}/{total}")

def val_ar(model, dataloader, device, decode):
  """ 
  Evaluation: autoregressive style
  """
  model.to(device)
  model.eval()
  total = 0
  num_correct = 0
  for idx, (batch_x, batch_y) in enumerate(dataloader):
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    question = batch_x[:, :equal_index+1]
    B, L = question.shape
    out = torch.zeros(batch_y.shape).to(device)
    for i in range(batch_y.shape[1]):
      logits = model.forward(question)[:, [-1], :] # (B, 1, vocab_size)
      logits = torch.argmax(logits, dim=-1) # (B, 1)
      question = torch.cat([question, logits], dim=1)
      out[:, i] = logits.squeeze() # (B,)
    matching_output = torch.all(out == batch_y, dim=1)
    num_correct += torch.sum(matching_output).item()
    total += batch_x.shape[0]
  print(f"Autoregressive inference result: {num_correct}/{total}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
  data_dir = "data/3_digits_addition_padded.txt"
  vocab_size, encode, decode, train_dataloader, val_dataloader = get_dataloader(data_dir, mode="train", batch_size=512)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("vocabe size:", vocab_size)
  
  # Build model
  model_name = "AT_220K"
  params = {"vocab_size": vocab_size,
          "context_length": 14,
          "model_size": 96,
          "num_heads": 8,
          "num_blocks": 2,
          "device": device}
  learning_rate = 1e-3
  
  # train model 1 - arithmatic transformer
  model1 = arithmaticTransformer(**params)
  print(f'Model {model_name} has {count_parameters(model1):,} trainable parameters')
  print(params)
  optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate)
  scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1000, gamma=0.5)
  train(model1, train_dataloader, optimizer1, scheduler1, device, max_iter=1e6, report_interval=50, model_name=model_name)
  val_tf(model1, val_dataloader, device, decode)


if __name__ == "__main__":
  main()
