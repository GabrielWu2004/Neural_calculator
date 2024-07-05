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
from train import count_parameters

def main():
  vocab_size, encode, decode, dataloader = get_dataloader("data/eval_data_100.txt", mode="eval", batch_size=None, max_length=10)
  device = 'cpu'
  final_model_path = "model/model_100kD_253kP.pth"
  model = torch.load(final_model_path).to(device)
  model.device = device
  print(f"The model has {count_parameters(model):,} trainable parameters")
  model.eval()
  total = 0
  correct = 0
  for batch_x, batch_y in dataloader:
    total +=1 
    print(decode(batch_x.tolist()[0]).strip())
    model_output = decode(model.generate_padded(batch_x, encode).tolist()[0][:-1])
    true_output = decode(batch_y.tolist()[0][:-1])
    print("True output:", true_output)
    print("model output:", model_output)
    print()
    if model_output == true_output:
      correct += 1
  print(f"score: {correct}/{total}")

if __name__ == "__main__":
  main()