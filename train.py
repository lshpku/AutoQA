import torch
import torch.nn as nn
from torch import optim
import torch.functional as F
from torch.utils.data import DataLoader
from model import Embedding, Discriminator
from utils import WordDict, QADataset
import numpy as np
from PIL import Image
import random
import time

lr = 0.01
betas = (0.5, 0.999)
BATCH_SIZE = 4
VEC_SIZE = 300
SHOW_EVERY = 1
EVAL_EVERY = 2000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = QADataset('train-words.pkl').to(device)
dictionary = WordDict('word-list.pkl')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
evalset = QADataset('valid-words.pkl').to(device)
evalset = DataLoader(evalset, batch_size=1, shuffle=True, num_workers=0)

print('dict_size:', len(dictionary))
print('dataset_size:', len(dataset))
print('evalset_size:', len(evalset))

embd, disc = None, None
embd_optim, disc_optim = None, None

LABEL = torch.LongTensor([0]).to(device)
hidden = torch.zeros(1, 1, VEC_SIZE).to(device)


def train_epoch():
    loss_t = 0

    for i, data in enumerate(dataloader):
        ques, real, fakes = data[0], data[1], data[2:]

        embd_optim.zero_grad()
        disc_optim.zero_grad()

        ques = ques.view(-1, 1)
        ques = embd(ques)

        real = real.view(-1, 1)
        results = [disc(ques, embd(real), hidden)]

        for j in fakes:
            j = j.view(-1, 1)
            results.append(disc(ques, embd(j), hidden))

        loss = disc.loss(torch.cat(results, 1), LABEL)
        loss.backward()
        embd_optim.step()
        disc_optim.step()
        loss_t += loss

        if (i+1) % SHOW_EVERY == 0:
            log = '{}\t{:.8f}\t{:.1f}%'.format(i+1, loss_t.item()/SHOW_EVERY,
                                               (i+1)*100/len(dataloader))
            print(log)
            with open('loss.txt', 'a') as f:
                f.write(log+'\n')
            loss_t = 0
        if (i+1) % EVAL_EVERY == 0:
            online_eval(0)


def online_eval(limit=None):
    loss_t = 0
    limit = len(evalset) if not limit else limit
    all_right = 0

    for i, data in enumerate(evalset):
        if i >= limit:
            break
        ques, real, fakes = data[0], data[1], data[2:]
        ques = ques.view(-1, 1)
        real = real.view(-1, 1)
        with torch.no_grad():
            ques = embd(ques)
            results = [disc(ques, embd(real), hidden)]
            for j in fakes:
                j = j.view(-1, 1)
                results.append(disc(ques, embd(j), hidden))
            results = torch.cat(results, 1)
            loss = disc.loss(results, LABEL)
        loss_t += loss
        _, topk = results.topk(1)
        if topk.item() == 0:
            all_right += 1

    log = '{}\t{}'.format(loss_t.item()/limit, all_right/limit)
    print(log)
    with open('evaluate.txt', 'a') as f:
        f.write(log+'\n')


def test_input():
    while True:
        sent = input('s2v> ')
        sent = dictionary[sent]
        sent = torch.LongTensor(sent)
        with torch.no_grad():
            result = torch.sigmoid(embd(sent, hidden))
        result = np.array(result).reshape(10, 30)*255
        img = Image.fromarray(result.astype(np.uint8))
        img = img.resize((300, 100))
        img.show()


def train(from_version=1, epoches=50):
    global embd, disc, embd_optim, disc_optim
    embd = Embedding(len(dictionary), VEC_SIZE).load(
        '.', from_version, device)
    disc = Discriminator(VEC_SIZE).load('.', from_version, device)
    embd_optim = optim.SGD(embd.parameters(), lr=lr)
    disc_optim = optim.SGD(disc.parameters(), lr=lr)
    for i in range(epoches):
        train_epoch()
        embd.save('.', i+from_version)
        disc.save('.', i+from_version)


train()
