import os
import torch
import torch.nn as nn
import torch.functional as F


class BasicModel(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def save(self, root='.', version=1):
        path = self.path(root, version)
        torch.save(self.state_dict(), path)
        print('save to', path)

    def load(self, root='.', version=1, device='cpu'):
        path = self.path(root, version)
        self.to(device)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            print('load pre-trained', path)
        return self

    def path(self, root: str, version: int):
        name = '{}-{:04d}.pth'.format(self.name, version)
        return os.path.join(root, name)


class SentenceEncoder(BasicModel):
    '''
    Encode sentences into vectors.
    '''

    def __init__(self, dict_size: int, vec_size: int):
        super().__init__('encd')
        self.embd = nn.Embedding(dict_size, vec_size)
        self.gru = nn.GRU(input_size=vec_size, hidden_size=vec_size)

    def forward(self, sentence, hidden):
        embedded = self.embd(sentence.view(-1, 1))
        output, _ = self.gru(embedded, hidden)
        return output[-1]


class Classifier(BasicModel):
    '''
    Classify between `k` catagories.\n
    Only for pre-train.
    '''

    def __init__(self, vec_size: int, catagories: int):
        super().__init__('clsf')
        self.linear = nn.Sequential(
            nn.Linear(vec_size, vec_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(vec_size//2, catagories),
            nn.LogSoftmax(),
        )
        self.loss = nn.NLLLoss()

    def forward(self, sentence):
        return self.linear(sentence)


class Discriminator(BasicModel):
    '''
    Discriminate between real and fake answers.\n
    Count loss only for each group.
    '''

    def __init__(self, vec_size: int):
        super().__init__('disc')
        self.linear = nn.Sequential(
            nn.Linear(vec_size * 2, vec_size),
            nn.ReLU(inplace=True),
            nn.Linear(vec_size, 1),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, question, answer):
        x = torch.cat((question, answer), 1)
        return self.linear(x)

'''
encoder = SentenceEncoder(3597, 128)
state = encoder.state_dict()
rnn = torch.load('rnn-0002.pth', map_location='cpu')
state.update({'embd.weight':rnn['embd.weight']})
encoder.load_state_dict(state)
encoder.save()
encoder.load()
'''