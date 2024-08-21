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
from train import count_parameters, val, val_iter

def eval_single():
  vocab_size, encode, decode, dataloader = get_dataloader("data/3_digits_eval_100.txt", mode="test", batch_size=1)
  device = 'cpu'
  final_model_path = "model/AT_1M.pth"
  model = torch.load(final_model_path).to(device)
  model.device = device
  print(f"The model has {count_parameters(model):,} trainable parameters")
  model.eval()
  total = 0
  correct = 0
  for idx, (batch_x, batch_y) in enumerate(dataloader):
    total +=1 
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    print(decode(batch_x.tolist()[0]).strip())
    model_output = decode(model.generate(batch_x, encode).tolist()[0][:-1])
    true_output = decode(batch_y.tolist()[0][:-1])
    print("True output:", true_output)
    print("model output:", model_output)
    print()
    if model_output == true_output:
      correct += 1
  print(f"score: {correct}/{total}")



if __name__ == "__main__":
  eval_single()