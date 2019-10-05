"""
Generate random data for pointer network
"""
import torch
from torch.utils.data import Dataset


def sample(min_length=5, max_length=12):
  """
  Generates a single example for a pointer network. The example consist in a tuple of two
  elements. First element is an unsorted array and the second element 
  is the result of applying argsort on the first element
  """
  array_len = torch.randint(low=min_length, 
                            high=max_length + 1,
                            size=(1,))
  x = torch.randint(high=array_len.item(), size=(array_len,))
  return x, x.argsort()


def batch(batch_size, min_len=5, max_len=12):
  array_len = torch.randint(low=min_len, 
                            high=max_len + 1,
                            size=(1,))

  x = torch.randint(high=10, size=(batch_size, array_len))
  return x, x.argsort(dim=1)

