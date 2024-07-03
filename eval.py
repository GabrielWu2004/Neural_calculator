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

def main():
  vocab_size, encode, decode, dataloader = get_dataloader("data/eval_data_100.txt", batch_size=None, mode="eval")
  device = 'cpu'
  final_model_path = 'model/final_model_340486.pth'
  model = torch.load(final_model_path).to(device)
  model.device = device
  model.eval()
  for batch_x, batch_y in dataloader:
    print(decode(batch_x.tolist()[0]))
    out = model.generate(batch_x, encode).tolist()[0]
    print("True output:", decode(batch_y.tolist()[0][:-1]))
    print("model output:", decode(out))

if __name__ == "__main__":
  main()