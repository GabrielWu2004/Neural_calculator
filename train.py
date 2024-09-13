import os
import argparse
from tqdm import tqdm
from model_architecture import arithmaticTransformer
from data_processing import tokenizer, streamingDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

# equal_index = 8
equal_index = 12
context_length = 19 
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


def train(model, dataloader, optimizer, scheduler, device, model_name, report_interval=100, max_iter=int(2e6), save_checkpoint=False, checkpoint_path=None, checkpoint_interval=None, resume_checkpoint=False, checkpoint_dir='model'):
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
  print("Training begin")
  with tqdm(enumerate(dataloader, start=start_iter), total=max_iter) as pbar:
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
      
      # Log result
      wandb.log({"loss": loss})
  
  # Save final model
  final_model_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
  torch.save(model, final_model_path)
  print(f"Final model saved to {final_model_path}")


def val_tf(model, dataloader, device, decode, verbose=0):
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
    if verbose:
      print(f"Batch {idx}: {torch.sum(matching_output).item()}/{batch_x.shape[0]}")
      if verbose == 2 and idx == 0:
        for item in range(logits.shape[0]):
          question = batch_x[item][:equal_index+1].tolist()
          print("Question:", decode(question))
          print("Model output:", decode(logits[item].tolist()))
          print("Correct answer:", decode(batch_y[item].tolist()))
          print()
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
  # argument parsing
  parser = argparse.ArgumentParser()
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--model_name", type=str, default="test")
  parser.add_argument("--model_size", type=int, default=256)
  parser.add_argument("--num_heads", type=int, default=8)
  parser.add_argument("--num_blocks", type=int, default=8)
  parser.add_argument("--lr_step_size", type=int, default=1000)
  parser.add_argument("--lr_gamma", type=float, default=0.98)
  args = parser.parse_args()
  
  # Load dataset
  training_data_dir = "data/complex_arithmetic_train"
  testing_data_dir = "data/complex_arithmetic_test"
  reverse = True
  vocab_size, encode, decode = tokenizer(os.path.join(training_data_dir, os.listdir(training_data_dir)[0]))
  train_dataset = streamingDataset(training_data_dir, encode=encode, reverse=reverse)
  train_dataloader = DataLoader(train_dataset, batch_size=1024)
  test_dataset = streamingDataset(testing_data_dir, encode=encode, reverse=reverse)
  test_dataloader = DataLoader(test_dataset, batch_size=64)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("vocabe size:", vocab_size)
  
  # Build model
  model_name = args.model_name
  params = {"vocab_size": vocab_size,
          "context_length": context_length,
          "model_size": args.model_size,
          "num_heads": args.num_heads,
          "num_blocks": args.num_blocks,
          "device": device}
  learning_rate = args.lr
  lr_step_size = args.lr_step_size # step decay
  lr_gamma = args.lr_gamma # step decay factor
  
  # Set up wandb logging
  wandb.init(
    project="arithmetic transformer",
    config={"model_name": model_name,
            "learning_rate": learning_rate,
            "lr_step_size": lr_step_size,
            "lr_gamma": lr_gamma,
            "reverse": reverse,
            **params,
            })
  
  # Model training
  model = arithmaticTransformer(**params)
  print(f'Model {model_name} has {count_parameters(model):,} trainable parameters')
  print(params)
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
  train(model, train_dataloader, optimizer, scheduler, device, max_iter=int(2e4), report_interval=50, model_name=model_name)
  val_tf(model, test_dataloader, device, decode, verbose=0)


if __name__ == "__main__":
  main()
