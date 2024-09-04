import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split

################## Data generation and tokenization ##################

def generate_data_complex(dest, num_digits=3, num_operands=3, num_samples=int(1e7)):
  """
  Geenerate strings of complex arithmetic expressions with addition and subtraction.
  Number of operants can be arbitrarily specified
  Front and back delimitor "$" included.
  Input padded to "num_digits" digits. Output padded to "num_digits + 1" digits with leading sign.
  Each data file contains 1M entries maximum. Multiple data files created to avoid github size constraint.
  """
  
  file_index = 0
  data_count = 0
  current_file = f"{dest}_{file_index}.txt"
  operators = ["+", "-"]
  
  with open(current_file, "w") as f:
    for i in range(num_samples):
      # Randomly generate operands and operations
      operands = np.random.randint(10**(num_digits-1), 10**num_digits, size=num_operands)
      ops = random.choices(operators, k=num_operands-1)
      
      # Generate equation string
      equation_str = str(operands[0])
      for j in range(len(ops)):
        equation_str += ops[j]
        equation_str += str(operands[j+1])
      result = eval(equation_str)
      zero_padding = (num_digits + 1 - len(str(np.abs(result)))) * "0"
      if result >= 0:
        equation_str = "$" + equation_str + "=+" + zero_padding + str(result) + "$"
      else:
        equation_str = "$" + equation_str + "=-" + zero_padding + str(np.abs(result)) + "$"
    
      # Write to dest
      f.write(f"{equation_str}\n")
      if (i+1)%1000000 == 0:
        file_index += 1
        current_file = f"{dest}_{file_index}.txt"
        print(f"new file created: {current_file}")
        f = open(current_file, "w")
    
    print(f"Data generation completed: {file_index + 1} files created.")


def tokenizer(dest):
  """
  Returns vocab_size, encode function, and decode function
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


################## Data generation and tokenization ##################

def plot_length_distribution(length_counts):
  lengths = list(length_counts.keys())
  counts = list(length_counts.values())
  plt.figure(figsize=(10, 5))
  plt.bar(lengths, counts)
  plt.xlabel('Line Length')
  plt.ylabel('Frequency')
  plt.title('Distribution of Line Lengths')
  plt.show()

def plot_digit_distribution(dir):
  """
  Plot digit distribution at each position directly from txt
  Currently only works for 3-digit additions
  """
  with open(dir, mode="r") as f:
    lines = f.readlines()
  first_line = lines[0]
  equal_index = first_line.find("=")
  digit_dict = collections.defaultdict(list) # index -> list of digits at that index
  digit_counts = collections.defaultdict(dict) # index -> dictionary counter of digits at that index
  
  for idx in range(equal_index+1, equal_index+5):
    for line in lines:
      digit_dict[idx].append(line[idx])
    digit_counts[idx] = collections.Counter(digit_dict[idx])

  # print(digit_counts[9]) 
  fig, axs = plt.subplots(2, 2, figsize=(12, 9))
  fig.suptitle('Target Token distribution')
  axs = axs.flatten()
  for i, (key, sub_dict) in enumerate(digit_counts.items()):
    keys = list(sub_dict.keys())
    values = list(sub_dict.values())
    axs[i].bar(keys, values)
    axs[i].set_title(f'Distribution of token {key}')
    axs[i].set_xlabel('Keys')
    axs[i].set_ylabel('Counts')
    axs[i].set_xticks(list(range(10)))  # Set the x-ticks to 0 through 13
    axs[i].set_xticklabels(list(range(10)))  # Set the x-tick labels to 0 through 13
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show()

################## Dataset and Dataloader ##################

class trainingDataset(Dataset):
  def __init__(self, dir, encode):
    self.x = []
    self.y = []
    equal_token = encode("=")[0]
    with open(dir, "r", encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        tokenized_line = encode(line)
        equal_index = tokenized_line.index(equal_token)
        self.x.append(tokenized_line[:-2])
        self.y.append(tokenized_line[equal_index+1:-1])
  
  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class testDataset(Dataset):
  def __init__(self, data_dest, encode):
    self.x = []
    self.y = []
    with open(data_dest, "r", encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        question, answer = line.strip().split("=")
        self.x.append(encode(question + "="))
        self.y.append(encode(answer))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def get_dataloader(dir, mode, batch_size, shuffle=True):
  """
  mode (str): "train", "test"
  Returns: vocab_size, encode, decode, dataloader (if "train", then return both train_ and val_dataloader)
  """
  vocab_size, encode, decode = tokenizer(dir)
  if mode == "test":
    dataset = testDataset(dir, encode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return vocab_size, encode, decode, dataloader
  elif mode == "train":
    dataset = trainingDataset(dir, encode)
    train_size = int(0.9*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return vocab_size, encode, decode, train_dataloader, val_dataloader
  else:
    raise NameError("Not a valid mode. Mode can be either 'train' or 'test'.")
  

def display_tokenizer(vocab_size, decode):
  for i in range(vocab_size):
    print(f"Token: {i}. char: {decode([i])}")


################## main functions ##################

def main_analyze_data():
  vocab_size, encode, decode = tokenizer("data/training_data_100k.txt")
  for i in range(vocab_size):
    print(f"Token: {i}. char: {decode([i])}")
  # training_dataset = trainingDataset("data/training_data_100k.txt", encode)
  # length_to_indices = group_by_length(training_dataset)
  # token_counters = token_by_length(length_to_indices, training_dataset)
  # plot_token_distribution(token_counters, vocab_size)


def main_generate_data():
  data_dir = "data/complex_arithmetic/3_operands_mix"
  generate_data_complex(data_dir)
  # vocab_size, encode, decode, train_dataloader, val_dataloader = get_dataloader(data_dir, mode="train", batch_size=4, shuffle=False)
  # print("train")
  # for idx, (batch_x, batch_y) in enumerate(train_dataloader):
  #   print("batch", idx)
  #   print(batch_x.shape)
  #   print(batch_y.shape)
  #   print()
  #   if idx > 3:
  #     break
  # print("eval")
  # for idx, (batch_x, batch_y) in enumerate(val_dataloader):
  #   print("batch", idx)
  #   print(batch_x)
  #   print(batch_y)
  #   print()
  #   if idx > 3:
  #     break

  # vocab_size, encode, decode, test_dataloader = get_dataloader("data/3_digits_eval_100.txt", mode="test", batch_size=1, shuffle=False)
  # display_tokenizer(vocab_size, decode)
  # for idx, (batch_x, batch_y) in enumerate(test_dataloader):
  #   print("batch", idx)
  #   print(batch_x)
  #   print(batch_y)
  #   print()
  #   if idx > 3:
  #     break
  # pass


if __name__ == "__main__":
  # main_analyze_data()
  main_generate_data()
  pass