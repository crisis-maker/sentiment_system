import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRU_Net(nn.Module):
    def __init__(self, output_size, hidden_dim, n_layers, embedding_dim, batch_size, device, len_):
        super(GRU_Net, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(len_, embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.h0 = torch.zeros(n_layers, batch_size, hidden_dim)
        # self.dropout = nn.Dropout(0.5)

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.out = output_size
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        x = x.long()
        x = self.embedding(x)
        # x = x.transpose(1, 2)
        # print(x.shape)

        out, hidden = self.gru(x, hidden)
        # print(out.shape)
        out = out.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(out)
        out = self.fc(out)
        # print(out.shape)

        out = out.view(batch_size, -1)
        l = []
        for i in range(self.out):
            l.append(out.shape[1] - 1 - i)
        out = out[:, l]

        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden