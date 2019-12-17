#from torch.utils.data import Dataset, DataLoader
import pickle
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WordDict():
    def __init__(self, path: str):
        # Special tags
        self.START = 0
        self.END = 1
        self.UNKNOWN = 2

        # Load word dict
        with open(path, 'r', encoding='utf-8') as f:
            word_list = f.read()
        word_list = word_list.split('\n')
        word_list = [i.split('\t')[1] for i in word_list if i]
        self.__dict = {i: j+3 for j, i in enumerate(word_list)}
        self.__list = ['<S>', '<E>', '<U>'] + word_list

    def __len__(self):
        return len(self.__dict) + 3

    def __getitem__(self, word):
        if isinstance(word, str):
            if word in self.__dict:
                return self.__dict[word]
            return self.UNKNOWN
        return [self[i] for i in word]  # recurse

    def restore(self, number):
        if isinstance(number, int):
            if number >= 0 and number < len(self.__list):
                return self.__list[number]
            return '<X>'
        return [self.restore(i) for i in number]  # recurse


class QADataset():
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            self.__items = pickle.load(f)

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, index):
        return self.__items[index]


class QACharDataset():
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            items = pickle.load(f)
        self.items = [i for i in items
                      if len(i[1]) == 1 and len(i[2]) >= 1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ques, real, fakes = self.items[index]
        data = [torch.LongTensor(ques), torch.LongTensor(real[0])]
        data += [torch.LongTensor(i) for i in fakes]
        return data


class NameDataset(Dataset):
    def __init__(self):
        path = os.path.join('torchdata', 'names')
        files = os.listdir(path)
        self.names = []
        self.types = len(files)
        for i, j in enumerate(files):
            with open(os.path.join(path, j), 'r', encoding='utf-8') as f:
                names = f.read().split('\n')
                self.names += [(self.__ord(j), i) for j in names if j]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name, tag = self.names[index]
        return torch.LongTensor(name), tag

    @staticmethod
    def __ord(name):
        return [ord(i) for i in name]


class CharDict():
    def __init__(self):
        with open('valchr.json', 'r') as f:
            chars = json.load(f)
        self.__n2c = chars
        self.__c2n = {j: i for i, j in enumerate(chars)}
        self.UNKNOWN = len(chars)

    def __len__(self):
        return self.UNKNOWN + 1

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= 0 and index < self.UNKNOWN:
                return self.__n2c[index]
            return '<UNK>'
        if isinstance(index, str) and len(index) <= 1:
            if not index:
                raise Exception('CharDict Error: empty index: ' + index)
            index = self.__strQ2B(index)
            if index in self.__c2n:
                return self.__c2n[index]
            return self.UNKNOWN
        return [self[i] for i in index]

    @staticmethod
    def __strQ2B(ustr: str) -> str:
        '''
        Double-byte to single-byte characters.
        '''
        rstr = []
        for uchar in ustr:
            code = ord(uchar)
            if code == 12288:
                inside_code = 32
            elif code >= 65281 and code <= 65374:
                code -= 65248
            rstr.append(chr(code))
        return ''.join(rstr)


class SogouDataset(Dataset):
    def __init__(self):
        files = [i for i in os.listdir('sogou') if i.endswith('.txt')]
        self.CATAGORIES = len(files)
        self.sents = []
        for i, j in enumerate(files):
            with open(os.path.join('sogou', j), 'r', encoding='utf-8') as f:
                text = f.read()
            sents = text.split('。')
            sents = [k+'。' for k in sents if k]
            self.sents += [(k, i) for k in sents]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent, tag = self.sents[index]
        sent = torch.LongTensor(char_dict[sent])
        tag = torch.LongTensor([tag])
        return sent, tag


class THUCDataset(Dataset):
    def __init__(self):
        path = 'THUC-slim'
        files = [i for i in os.listdir(path) if i.endswith('.txt')]
        self.CATAGORIES = len(files)
        self.sents = []
        for i, j in enumerate(files):
            with open(os.path.join(path, j), 'r', encoding='utf-8') as f:
                text = f.read()
            sents = text.split('。')
            sents = [k+'。' for k in sents if k]
            self.sents += [(k, i) for k in sents]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent, tag = self.sents[index]
        sent = torch.LongTensor(char_dict[sent])
        tag = torch.LongTensor([tag])
        return sent, tag


char_dict = CharDict()
