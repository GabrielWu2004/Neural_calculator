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

def generate_data_pad(num_samples, upper_bound, dest, max_length, pad):
  """ 
  Generate equations with only two operands
  pad (str): either "front" or "back"
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
      if len(string) < max_length:
        if pad == "front":
          string = " "*(max_length - len(string)) + string
        else:
          string = string + " "*(max_length - len(string))
      f.write(f"{string}\n")

def tokenizer(dest):
  """
  Returns vocab_size, encode, and decode
  """
  with open(dest, "r", encoding='utf-8') as f:
    text = f.read()
  char = sorted(list(set(text))+ [' '])
  vocab_size = len(char)
  char_to_int = {c: i for i, c in enumerate(char)}
  int_to_char = {i: c for i, c in enumerate(char)}
  encode = lambda s: [char_to_int[c] for c in s] # string to list
  decode = lambda l: ''.join([int_to_char[i] for i in l]) # list to string
  return vocab_size, encode, decode

def analyze_line_lengths(dir):
  """
  Returns a counter object of the count of each token length
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

def plot_token_distribution(token_counters, vocab_size):
  fig, axs = plt.subplots(3, 2, figsize=(12, 12))
  fig.suptitle('Target Token distribution')
  axs = axs.flatten()
  x_values = list(range(vocab_size))
  for i, (key, sub_dict) in enumerate(token_counters.items()):
    keys = list(sub_dict.keys())
    values = list(sub_dict.values())
    axs[i].bar(keys, values)
    axs[i].set_title(f'Distribution of token {key}')
    axs[i].set_xlabel('Keys')
    axs[i].set_ylabel('Counts')
    axs[i].set_xticks(x_values)  # Set the x-ticks to 0 through 13
    axs[i].set_xticklabels(x_values)  # Set the x-tick labels to 0 through 13

  # Adjust layout
  plt.tight_layout(rect=[0, 0, 1, 0.96])
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

def token_by_length(length_to_indices, dataset):
  """
  Input: a dictionary whose keys are lengths and values are lists of indices
  Return: a dictionary of counters, where keys are lengths and values are target token counter
  """
  token_counters = collections.defaultdict(collections.Counter)
  for length, indices in length_to_indices.items():
    targets = [dataset[idx][1].tolist() for idx in indices]
    token_counters[length] = collections.Counter(targets)
  return token_counters

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
  

class trainingDatasetPadded(Dataset):
  def __init__(self, dir, encode, max_length):
    self.x = []
    self.y = []
    equal_token = encode("=")[0]
    newline_token = encode("\n")[0]
    padding_token = encode(" ")[0]
    with open(dir, "r", encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        tokenzied_line = encode(line)
        equal_index = tokenzied_line.index(equal_token)
        newline_index = tokenzied_line.index(newline_token)
        for i in range(equal_index+1, newline_index+1):
          self.x.append([padding_token]*(max_length-i) + tokenzied_line[:i])
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
  def __init__(self, data_dest, encode, max_length):
    self.x = []
    self.y = []
    with open(data_dest, "r", encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        question, answer = line.split("=")
        # Two lines added for padding
        cur_length = len(question)
        self.x.append(encode(" "*(max_length-cur_length-1) + question + "="))
        self.y.append(encode(answer))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    if isinstance(idx, list):
      batch_x = torch.tensor([self.x[i] for i in idx])
      batch_y = torch.tensor([self.y[i] for i in idx])
      return batch_x, batch_y
    return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def get_dataloader(dir, mode, batch_size, max_length, shuffle=True):
  """
  mode (str): "eval", "train_single", "train_padded", "train_bucket"
  Returns: vocab_size, encode, decode, dataloader
  """
  vocab_size, encode, decode = tokenizer(dir)
  if mode == "eval":
    dataset = evalDataset(dir, encode, max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
  elif mode == "train_single":
    dataset = trainingDataset(dir, encode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
  elif mode == "train_padded":
    dataset = trainingDatasetPadded(dir, encode, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  elif mode == "train_bucket":
    dataset = trainingDataset(dir, encode)
    length_to_indices = group_by_length(dataset)
    batch_sampler = bucketSampler(length_to_indices, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=custom_collate_fn)
  else:
    raise NameError("Not a valid mode")
  return vocab_size, encode, decode, dataloader


def main_analyze_data():
  vocab_size, encode, decode = tokenizer("data/training_data_100k.txt")
  for i in range(vocab_size):
    print(f"Token: {i}. char: {decode([i])}")
  # training_dataset = trainingDataset("data/training_data_100k.txt", encode)
  # length_to_indices = group_by_length(training_dataset)
  # token_counters = token_by_length(length_to_indices, training_dataset)
  # plot_token_distribution(token_counters, vocab_size)


def main_generate_data():
  # generate_data_simple(1_000_000, 100, "data/training_data_1M.txt")
  # generate_data_simple(100, 100, "data/eval_data_100.txt")
  generate_data_pad(100_000, 100, "data/training_data_pad_100K.txt", 10, "front")
  # length_count = analyze_line_lengths("data/training_data_100k.txt")
  # plot_length_distribution(length_count)
  
  vocab_size, encode, decode, train_dataloader = get_dataloader("data/training_data_1k.txt", mode="train_padded", batch_size=2, max_length=10)
  for idx, (batch_x, batch_y) in enumerate(train_dataloader):
    print("batch", idx)
    print(batch_x)
    print(batch_y)
    if idx > 10:
      break
  # a = torch.tensor(1).tolist()
  # print(type(a))
  
  pass


if __name__ == "__main__":
  # main_analyze_data()
  main_generate_data()
  pass