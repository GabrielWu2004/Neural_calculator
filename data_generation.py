import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

################## Useful Functions ##################

def generate_data_simple(num_samples, upper_bound, dest):
  """ 
  Generate equations with only two operands
  """
  with open(dest, "w") as f:
    for i in range(num_samples):
      num1 = np.random.randint(0, upper_bound)
      num2 = np.random.randint(0, upper_bound)
      opt = np.random.randint(0, 2)
      if opt == 0:
        res = num1 + num2
        string = f"{num1}+{num2}={res}"
      else:
        res = num1 - num2
        string = f"{num1}-{num2}={res}"
      f.write(f"{string}\n")

def tokenizer(dest):
  """
  Returns vocab_size, encode, and decode
  """
  with open(dest, "r", encoding='utf-8') as f:
    text = f.read()
  char = sorted(list(set(text)))
  vocab_size = len(char)
  char_to_int = {c: i for i, c in enumerate(char)}
  int_to_char = {i: c for i, c in enumerate(char)}
  encode = lambda s: [char_to_int[c] for c in s] # string to list
  decode = lambda l: ''.join([int_to_char[i] for i in l]) # list to string
  return vocab_size, encode, decode

def analyze_line_lengths(dir):
  """
  Returns a counter object
  """
  with open(dir, 'r') as file:
    line_lengths = [len(line) for line in file]
  length_counts = collections.Counter(line_lengths)
  return length_counts

def plot_length_distribution(length_counts):
  lengths = list(length_counts.keys())
  counts = list(length_counts.values())
  plt.figure(figsize=(10, 5))
  plt.bar(lengths, counts)
  plt.xlabel('Line Length')
  plt.ylabel('Frequency')
  plt.title('Distribution of Line Lengths')
  plt.show()

def group_by_length(dataset):
  """
  Input: pytorch dataset
  Output: collections.defaultdict
  """
  length_to_indices = collections.defaultdict(list)
  for idx, (x, y) in enumerate(dataset):
    length_to_indices[x.shape[0]].append(idx)
  return length_to_indices

def custom_collate_fn(batch):
    batch_x, batch_y = zip(*batch)
    batch_x = torch.stack(batch_x, dim=0)
    batch_y = torch.stack(batch_y, dim=0)
    return batch_x, batch_y


################## Dataset and Dataloader ##################

class bucketSampler(torch.utils.data.Sampler):
  def __init__(self, length_to_indices, batch_size):
    self.length_to_indices = length_to_indices
    self.batch_size = batch_size
    self.buckets = list(length_to_indices.values()) # a list of buckets, each one is a list
  
  def __iter__(self):
    """
    Yield a batch of the uniform size
    """
    for bucket in self.buckets:
      random.shuffle(bucket)
      for i in range(0, len(bucket), self.batch_size):
        if (i + self.batch_size > len(bucket)):
          yield bucket[i:]
        else:
          yield bucket[i:i+self.batch_size]
  
  def __len__(self):
    return sum(len(bucket) // self.batch_size + (1 if len(bucket) % self.batch_size != 0 else 0) for bucket in self.buckets)


class trainingDataset(Dataset):
  def __init__(self, dir, encode):
    self.x = []
    self.y = []
    equal_token = encode("=")[0]
    newline_token = encode("\n")[0]
    with open(dir, "r", encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        tokenzied_line = encode(line)
        equal_index = tokenzied_line.index(equal_token)
        newline_index = tokenzied_line.index(newline_token)
        for i in range(equal_index+1, newline_index+1):
          self.x.append(tokenzied_line[:i])
          self.y.append(tokenzied_line[i])

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    if isinstance(idx, list):
      batch_x = torch.tensor([self.x[i] for i in idx])
      batch_y = torch.tensor([self.y[i] for i in idx])
      return batch_x, batch_y
    return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class evalDataset(Dataset):
  def __init__(self, data_dest, encode):
    self.x = []
    self.y = []
    with open(data_dest, "r", encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        question, answer = line.split("=")
        self.x.append(encode(question + "="))
        self.y.append(encode(answer))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    if isinstance(idx, list):
      batch_x = torch.tensor([self.x[i] for i in idx])
      batch_y = torch.tensor([self.y[i] for i in idx])
      return batch_x, batch_y
    return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def get_dataloader(dir, batch_size, mode="train"):
  """
  Returns: vocab_size, encode, decode, dataloader
  """
  vocab_size, encode, decode = tokenizer(dir)
  if mode == "eval":
    dataset = evalDataset(dir, encode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return vocab_size, encode, decode, dataloader
  elif mode == "train_single":
    dataset = trainingDataset(dir, encode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return vocab_size, encode, decode, dataloader
  dataset = trainingDataset(dir, encode)
  length_to_indices = group_by_length(dataset)
  batch_sampler = bucketSampler(length_to_indices, batch_size)
  dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=custom_collate_fn)
  return vocab_size, encode, decode, dataloader


if __name__ == "__main__":
  # generate_data_simple(1_000_000, 100, "data/training_data_1M.txt")
  # generate_data_simple(100, 100, "data/eval_data_100.txt")
  length_count = analyze_line_lengths("data/training_data_100k.txt")
  plot_length_distribution(length_count)
  # vocab_size, encode, decode = tokenizer("data/training_data_100k.txt")
  # training_dataset = trainingDataset("data/training_data_100k.txt", encode)
  # length_to_indices = group_by_length(training_dataset)
  vocab_size, encode, decode, train_dataloader = get_dataloader("data/training_data_1k.txt", batch_size=None, mode="train_single")
  for idx, (batch_x, batch_y) in enumerate(train_dataloader):
    print("batch", idx)
    print(batch_x.shape)
    print(batch_y.shape)
    if idx > 10:
      break
  pass