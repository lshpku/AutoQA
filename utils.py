#from torch.utils.data import Dataset, DataLoader
import pickle
import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WordDict():
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            word_list = pickle.load(f)
        self.num2word = word_list
        self.word2num = {j: i for i, j in enumerate(word_list)}
        self.num2word.append('<UNK>')
        self.word2num['<UNK>'] = len(word_list)-1
        self.UNK = len(word_list)-1

    def __len__(self):
        return len(self.word2num)

    def __getitem__(self, content):
        if isinstance(content, str):
            if content in self.word2num:
                return self.word2num[content]
            return self.UNK
        if isinstance(content, int):
            if content >= 0 and content < len(self.num2word):
                return self.num2word[content]
            return '<UNK>'
        return [self[i] for i in content]  # recurse


class QADataset(Dataset):
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            items = pickle.load(f)
        self.items = [i for i in items if len(i[1]) and len(i[2])]
        self.device = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ques, real, fakes = self.items[index]
        if len(real) > 1:
            real = random.choice(real)
        data = [torch.LongTensor(ques), torch.LongTensor(real)]
        data += [torch.LongTensor(i) for i in fakes]
        if self.device:
            data = [i.to(self.device) for i in data]
        return data
    
    def to(self, device):
        self.device = device
        return self
