import torch
import torch.nn as nn
from utils import WordDict


class EmbeddingRNN(nn.Module):
    '''
    input: (`sentence_len`, `batch_size`, `input_size`)\n
    hidden: (`num_layers`, `batch_size`, `hidden_size`)
    '''

    def __init__(self, words: int):
        super().__init__()
        self.embd = nn.Embedding(words, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=64)

    def forward(self, word, hidden, end=False):
        embedded = self.embd(word).unsqueeze(1)
        output, hidden = self.gru(embedded, hidden)
        if end:
            return output[-1]
        return output, hidden


class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.Linear(input_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, question, answer):
        x = torch.cat((question, answer), 1)
        return self.linear(x)
