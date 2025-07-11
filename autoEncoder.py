import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(84*84, 128), # reduce N, 7068 -> N,128
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 12), 
            nn.ReLU(),
            nn.Linear(12, 3), # -> N,3 
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Sigmoid(),
           
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
