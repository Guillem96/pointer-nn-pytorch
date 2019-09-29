"""
Module implementing the pointer network proposed at: https://arxiv.org/abs/1506.03134

The implementation try to follows the variables naming conventions

ei: Encoder hidden state

di: Decoder hidden state
di_prime: Attention aware decoder state

W1: Learnable matrix (Attention layer)
W2: Learnable matrix (Attention layer)
V: Learnable parameter (Attention layer)

uj: Energy vector (Attention Layer)
aj: Attention mask over the input
"""
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import sample, batch

HIDDEN_SIZE = 256

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
EPOCHS = 10


class Encoder(nn.Module):
  def __init__(self, hidden_size: int):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
  
  def forward(self, x: torch.Tensor):
    # x: (BATCH, ARRAY_LEN, 1)
    return self.lstm(x)


class Attention(nn.Module):
  def __init__(self, hidden_size, units):
    super(Attention, self).__init__()
    self.W1 = nn.Linear(hidden_size, units, bias=False)
    self.W2 = nn.Linear(hidden_size, units, bias=False)
    self.V =  nn.Linear(units, 1, bias=False)

  def forward(self, 
              encoder_out: torch.Tensor, 
              decoder_hidden: torch.Tensor):
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # decoder_hidden: (BATCH, HIDDEN_SIZE)

    # Add time axis to decoder hidden state
    # in order to make operations compatible with encoder_out
    # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
    decoder_hidden_time = decoder_hidden.unsqueeze(1)

    # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
    # Note: we can add the both linear outputs thanks to broadcasting
    uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
    uj = torch.tanh(uj)

    # uj: (BATCH, ARRAY_LEN, 1)
    uj = self.V(uj)

    # Attention mask over inputs
    # aj: (BATCH, ARRAY_LEN, 1)
    aj = F.softmax(uj, dim=1)

    # di_prime: (BATCH, HIDDEN_SIZE)
    di_prime = aj * encoder_out
    di_prime = di_prime.sum(1)
    
    return di_prime, uj.squeeze(-1)
    

class Decoder(nn.Module):
  def __init__(self, 
               hidden_size: int,
               attention_units: int = 10):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(hidden_size + 1, hidden_size, batch_first=True)
    self.attention = Attention(hidden_size, attention_units)

  def forward(self, 
              x: torch.Tensor, 
              hidden: Tuple[torch.Tensor], 
              encoder_out: torch.Tensor):
    # x: (BATCH, 1, 1) 
    # hidden: (1, BATCH, HIDDEN_SIZE)
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # For a better understanding about hidden shapes read: https://pytorch.org/docs/stable/nn.html#lstm
    
    # Get hidden states (not cell states) 
    # from the first and unique LSTM layer 
    ht = hidden[0][0]  # ht: (BATCH, HIDDEN_SIZE)

    # di: Attention aware hidden state -> (BATCH, HIDDEN_SIZE)
    # att_w: Not 'softmaxed', torch will take care of it -> (BATCH, ARRAY_LEN)
    di, att_w = self.attention(encoder_out, ht)
    
    # Append attention aware hidden state to our input
    # x: (BATCH, 1, 1 + HIDDEN_SIZE)
    x = torch.cat([di.unsqueeze(1), x], dim=2)
    
    # Generate the hidden state for next timestep
    _, hidden = self.lstm(x, hidden)
    return hidden, att_w


class PointerNetwork(nn.Module):
  def __init__(self, 
               encoder: nn.Module, 
               decoder: nn.Module):
    super(PointerNetwork, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, 
              x: torch.Tensor, 
              y: torch.Tensor, 
              teacher_force_ratio=.5):
    # x: (BATCH_SIZE, ARRAY_LEN)
    # y: (BATCH_SIZE, ARRAY_LEN)

    # Array elements as features
    # encoder_in: (BATCH, ARRAY_LEN, 1)
    encoder_in = x.unsqueeze(-1).type(torch.float)

    # out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # hs: tuple of (NUM_LAYERS, BATCH, HIDDEN_SIZE)
    out, hs = encoder(encoder_in)

    # Accum loss throughout timesteps
    loss = 0

    # Save outputs at each timestep
    # outputs: (ARRAY_LEN, BATCH)
    outputs = torch.zeros(out.size(1), out.size(0), dtype=torch.long)
    
    # First decoder input is always 0
    # dec_in: (BATCH, 1, 1)
    dec_in = torch.zeros(out.size(0), 1, 1, dtype=torch.float)
    
    for t in range(out.size(1)):
      hs, att_w = decoder(dec_in, hs, out)
      predictions = F.softmax(att_w, dim=1).argmax(1)

      # Pick next index
      # If teacher force the next element will we the ground truth
      # otherwise will be the predicted value at current timestep
      teacher_force = random.random() < teacher_force_ratio
      idx = y[:, t] if teacher_force else predictions
      dec_in = torch.stack([x[b, idx[b].item()] for b in range(x.size(0))])
      dec_in = dec_in.view(out.size(0), 1, 1).type(torch.float)

      # Add cross entropy loss (F.log_softmax + nll_loss)
      loss += F.cross_entropy(att_w, y[:, t])
      outputs[t] = predictions

    # Weight losses, so every element in the batch 
    # has the same 'importance' 
    batch_loss = loss / y.size(0)

    return outputs, batch_loss


def train(model, optimizer, epoch, clip=1.):
  """Train single epoch"""
  print('Epoch [{}] -- Train'.format(epoch))
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()

    # Forward
    x, y = batch(BATCH_SIZE)
    out, loss = model(x, y)

    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    if (step + 1) % 100 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))


@torch.no_grad()
def evaluate(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val = batch(4)
  
  out, _ = model(x_val, y_val, teacher_force_ratio=0.)
  out = out.permute(1, 0)

  for i in range(out.size(0)):
    print('{} --> {} --> {}'.format(
      x_val[i], 
      x_val[i].gather(0, out[i]), 
      x_val[i].gather(0, y_val[i])
    ))


encoder = Encoder(HIDDEN_SIZE)
decoder = Decoder(HIDDEN_SIZE)
ptr_net = PointerNetwork(encoder, decoder)

optimizer = optim.Adam(ptr_net.parameters())

for epoch in range(EPOCHS):
  train(ptr_net, optimizer, epoch + 1)
  evaluate(ptr_net, epoch + 1)
  