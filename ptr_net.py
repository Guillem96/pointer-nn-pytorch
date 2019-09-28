import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import sample, batch

HIDDEN_SIZE = 256

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
EPOCHS = 20


class Encoder(nn.Module):
  def __init__(self, hidden_size):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
  
  def forward(self, x):
    return self.lstm(x)


class Attention(nn.Module):
  def __init__(self, hidden_size, units):
    super(Attention, self).__init__()
    self.W1 = nn.Linear(hidden_size, units, bias=False)
    self.W2 = nn.Linear(hidden_size, units, bias=False)
    self.V =  nn.Linear(units, 1, bias=False)

  def forward(self, encoder_hidden, decoder_hidden):
    decoder_hidden_time = decoder_hidden.unsqueeze(1)
    uj = self.W1(encoder_hidden) + self.W2(decoder_hidden_time)
    uj = torch.tanh(uj)
    uj = self.V(uj)

    aj = F.softmax(uj, dim=1)

    di_prime = aj * encoder_hidden
    di_prime = di_prime.sum(1)
    
    return di_prime, uj
    

class Decoder(nn.Module):
  def __init__(self, hidden_size):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(hidden_size + 1, hidden_size, batch_first=True)
    self.attention = Attention(hidden_size, 10)

  def forward(self, x, hidden, encoder_out):
    di, att_w = self.attention(encoder_out, hidden[0][0])
    x = torch.cat([di.unsqueeze(1), x], dim=2)
    _, hidden = self.lstm(x, hidden)
    return hidden, att_w


class PointerNetwork(nn.Module):
  def __init__(self, encoder, decoder):
    super(PointerNetwork, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, x, y, teacher_force_ratio=.5):
    out, hs = encoder(x.unsqueeze(-1).type(torch.float))

    loss = 0

    outputs = torch.zeros(out.size(1), out.size(0), 1, dtype=torch.long)
    dec_in = torch.zeros(out.size(0), 1, 1, dtype=torch.float)
    
    for t in range(out.size(1)):
      hs, att_w = decoder(dec_in, hs, out)
      predictions = F.softmax(att_w, dim=1).argmax(1)

      # Pick next index
      # If teacher force the next element will we the ground truth
      # otherwise will be the predicted value at current timestep
      idx = y[:, t] if random.random() < teacher_force_ratio else predictions

      dec_in = torch.stack([x[b, idx[b].item()] for b in range(x.size(0))])
      dec_in = dec_in.view(out.size(0), 1, 1).type(torch.float)

      loss += F.cross_entropy(att_w.squeeze(-1), y[:, t])
      outputs[t] = predictions

    batch_loss = loss / y.size(0)

    return outputs.type(torch.long), batch_loss


def train(model, optimizer, epoch, clip=1.):
  print('Epoch [{}] -- Train'.format(epoch))
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()

    x, y = batch(BATCH_SIZE)
    out, loss = model(x, y)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), clip)

    if (step + 1) % 100 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))

    optimizer.step()


@torch.no_grad()
def evaluate(model, epoch):
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val = batch(4)
  
  out, _ = ptr_net(x_val, y_val)
  out = out.permute(1, 0, 2).squeeze(-1)

  for i in range(out.size(0)):
    print('{} --> {} --> {}'.format(
      x_val[i], x_val[i].gather(0, out[i]), x_val[i].gather(0, y_val[i])
    ))


encoder = Encoder(HIDDEN_SIZE)
decoder = Decoder(HIDDEN_SIZE)
ptr_net = PointerNetwork(encoder, decoder)

optimizer = optim.Adam(ptr_net.parameters())

for epoch in range(EPOCHS):
  train(ptr_net, optimizer, epoch + 1)
  evaluate(ptr_net, epoch + 1)
  