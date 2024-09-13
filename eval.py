import numpy as np
import os
from tqdm import tqdm
from model_architecture import arithmaticTransformer
from data_processing import testDataset, tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train import count_parameters

def eval(reverse=True):
  # Load data
  test_data_path = "data/complex_arithmetic_test/testing_data_big.txt"
  # test_data_path = "data/3_digit_addition_test/3_digits_eval_100.txt"
  vocab_size, encode, decode = tokenizer(test_data_path)
  test_dataset = testDataset(test_data_path, encode)
  test_dataloader = DataLoader(test_dataset, batch_size=1)
  
  # Load model
  device = 'cpu'
  final_model_path = "model/ATC_FINAL.pth"
  # final_model_path = "model/AT_FINAL.pth"
  model = torch.load(final_model_path).to(device)
  model.device = device
  print(f"The model has {count_parameters(model):,} trainable parameters")
  model.eval()
  
  # Evaluation
  total = 0
  correct = 0
  eval_log_dir = "eval_log/complex_arithmetic"
  with open(eval_log_dir, "w") as f:
    f.write(f"Model path: {final_model_path} \n")
    f.write(f"Number of parameters: {count_parameters(model):,} \n \n")
    for idx, (batch_x, batch_y) in enumerate(test_dataloader):
      total +=1 
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      print(decode(batch_x.tolist()[0][1:]).strip())
      out, prob = model.generate(batch_x, encode)
      out = out.tolist()[0][:-1]
      if reverse:
        model_output = decode(out[::-1])
      else:
        model_output = decode(out)
      true_output = decode(batch_y.tolist()[0][:-1])
      print(f"True output: {true_output}")
      print(f"model output: {model_output}. Confidence: {prob*100:.2f}%")
      print()
      f.write(f"{decode(batch_x.tolist()[0][1:]).strip()}\n")
      f.write(f"True output: {true_output} \n")
      f.write(f"model output: {model_output}. Confidence: {prob*100:.2f}% \n \n")
      if model_output == true_output:
        correct += 1
    print(f"score: {correct}/{total}")
    f.write(f"score: {correct}/{total}")



if __name__ == "__main__":
  eval()