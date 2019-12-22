import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def save(self, root='.', version=1):
        path = self.path(root, version)
        torch.save(self.state_dict(), path)
        print('save model to \"{}\"'.format(path))

    def load(self, root='.', version=1, device='cpu'):
        path = self.path(root, version)
        self.to(device)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            print('load pre-trained model \"{}\"'.format(path))
        else:
            print('init model \"{}\"'.format(self.name))
        return self

    def path(self, root: str, version: int):
        name = '{}-{:04d}.pth'.format(self.name, version)
        return os.path.join(root, name)


class Embedding(BasicModel):
    def __init__(self, dict_size: int, vec_size: int):
        super().__init__('embd')
        self.embd = nn.Embedding(dict_size, vec_size)

    def forward(self, sent):
        return self.embd(sent)


class Discriminator(BasicModel):
    def __init__(self, vec_size: int):
        super().__init__('disc')
        self.q_gru = nn.GRU(vec_size, vec_size)
        '''
        # attention used parameters
        self.attentionW = nn.Parameter(torch.randn(size=(vec_size, vec_size)))
        self.attentionV = nn.Parameter(torch.randn(size=(vec_size, 1)))
        '''
        self.a_gru = nn.GRU(vec_size*2, vec_size)
        self.linear = nn.Sequential(
            nn.Linear(vec_size, vec_size//2),
            nn.Linear(vec_size//2, 1),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, ques, answ, hidden):
        output, _ = self.q_gru(ques, hidden)
        ques_vector = output[-1]
        '''
        # attention: calculate weights for each output step
        att_weight_list = []
        N = output.size()[0]
        for i in range(N):
            x = torch.mm(output[i], self.attentionW)
            x = F.tanh(x)
            x = torch.mm(x, self.attentionV)
            att_weight_list.append(x)
        att_weight = att_weight_list[0]
        for i in range(1, N):
            att_weight = torch.cat([att_weight, att_weight_list[i]], dim = 1)
        att_weight = F.softmax(att_weight, dim=1)
        ques_vector = torch.matmul(att_weight, output.squeeze(1))
        '''
        ques = ques_vector.repeat(answ.size()[0], 1, 1)
        output, _ = self.a_gru(torch.cat((answ, ques), 2), hidden)
        result = self.linear(output[-1])
        return result


class ConvGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1, dilation)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 3, 1, 1, dilation)

    def forward(self, comp):
        direct = self.conv1(comp)
        gate = self.conv2(comp)
        return F.glu(torch.cat((direct, gate), 1), 1)


def load_embedding(path: str):
    with open(path, 'rb') as f:
        word_vec = pickle.load(f)
    dict_size, vec_size = len(word_vec), len(word_vec[0])
    print('{}(+1) {}'.format(dict_size, vec_size))
    word_vec = torch.FloatTensor(word_vec)
    word_vec = torch.cat((word_vec, torch.randn(1, 300)), 0)
    encoder = Embedding(dict_size+1, vec_size)
    state = encoder.state_dict()
    state.update({'embd.weight': word_vec})
    encoder.load_state_dict(state)
    encoder.save()
    encoder.load()


def check_embedding():
    from utils import WordDict
    from PIL import Image
    wd = WordDict('word-list.pkl')
    print('dict_size:', len(wd))
    encd = Embedding(249974, 300).load()
    while True:
        word = input('> ')
        num = wd[word]
        word = wd[num]
        print(num, word)
        with torch.no_grad():
            code = encd.embd(torch.LongTensor([num]))
        code = torch.sigmoid(code)
        code = code.numpy().reshape(10, 30)
        code = (code * 255).astype(np.uint8)
        Image.fromarray(code).resize((300, 100)).show()


if __name__ == '__main__':
    # load_embedding('word-vec.pkl')
    check_embedding()
