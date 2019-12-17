import torch
import torch.nn as nn
from torch import optim
import torch.functional as F
from torch.utils.data import DataLoader
from model import SentenceEncoder, Classifier, Discriminator
from utils import WordDict, QADataset, QACharDataset
from utils import THUCDataset, char_dict
import numpy as np
from PIL import Image
import random
import time

lr = 0.01
betas = (0.5, 0.999)
BATCH_SIZE = 4
VEC_SIZE = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dataset = THUCDataset()
#dataset = QADataset('train-words.pth')
dataset = QACharDataset('train-chars.pth')
dictionary = char_dict
#dictionary = WordDict('word-dict.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
evaluate = QACharDataset('valid-chars.pth')
evaluate = DataLoader(evaluate, batch_size=1, shuffle=True, num_workers=0)

print('dict_size:', len(dictionary))
print('dataset_size:', len(dataset))

encd, clsf, disc = None, None, None
encd_optim, clsf_optim, disc_optim = None, None, None

LABEL = torch.LongTensor([0]).to(device)
hidden = torch.zeros(1, 1, VEC_SIZE).to(device)


def pretrain_epoch():
    loss_t = 0
    batch = []

    for i, data in enumerate(dataloader):
        data = (data[0].to(device), data[1].to(device))
        batch.append(data)
        if len(batch) < BATCH_SIZE:
            continue

        encd_optim.zero_grad()
        clsf_optim.zero_grad()
        loss = 0

        for j in batch:
            name, tag = j
            tag = tag.view(-1)
            result = clsf(encd(name, hidden))
            loss += clsf.loss(result, tag)

        loss.backward()
        encd_optim.step()
        clsf_optim.step()
        loss_t += loss
        batch.clear()

        if i % 160 == 159:
            print('{}\t{:.8f}\t{:.1f}%'.format(i+1, loss_t.item()/160,
                                               (i+1)*100/len(dataloader)))
            loss_t = 0


def predict(limit=0):
    loss_t = 0
    limit = len(evaluate) if not limit else limit
    all_right = 0

    for i, data in enumerate(evaluate):
        if i >= limit:
            break
        ques, real, fakes = data[0], data[1], data[2:]
        ques = ques.to(device)
        real = real.to(device)
        with torch.no_grad():
            ques = encd(ques, hidden)
            results = [disc(ques, encd(real, hidden))]
            for j in fakes:
                j = j.to(device)
                results.append(disc(ques, encd(j, hidden)))
            results = torch.cat(results, 1)
            loss = disc.loss(results, LABEL)
        loss_t += loss
        _, topk = results.topk(1)
        if topk.item() == 0:
            all_right += 1

    with open('predict.txt', 'a') as f:
        f.write('{}\t{}'.format(loss_t.item()/limit, all_right/limit))


def train_epoch():
    loss_t = 0

    for i, data in enumerate(dataloader):
        ques, real, fakes = data[0], data[1], data[2:]

        encd_optim.zero_grad()
        disc_optim.zero_grad()

        ques = ques.to(device)
        ques = encd(ques, hidden)

        real = real.to(device)
        results = [disc(ques, encd(real, hidden))]

        for j in fakes:
            j = j.to(device)
            results.append(disc(ques, encd(j, hidden)))

        loss = disc.loss(torch.cat(results, 1), LABEL)
        loss.backward()
        encd_optim.step()
        disc_optim.step()
        loss_t += loss

        if i % 200 == 199:
            print('{}\t{:.8f}\t{:.1f}%'.format(i+1, loss_t.item()/200,
                                               (i+1)*100/len(dataloader)))
            loss_t = 0
        if i % 2000 == 1999:
            predict(200)


def train(from_version=1, epoches=50):
    global encd, disc, encd_optim, disc_optim
    encd = SentenceEncoder(len(dictionary), VEC_SIZE).load(
        '.', from_version, device)
    disc = Discriminator(VEC_SIZE).load('.', from_version, device)
    encd_optim = optim.SGD(encd.parameters(), lr=lr)
    disc_optim = optim.Adam(disc.parameters(), lr=lr, betas=betas)
    for i in range(epoches):
        train_epoch()
        encd.save('.', i+from_version)
        disc.save('.', i+from_version)


def pretrain(from_version=1, epoches=50):
    global encd, clsf, encd_optim, clsf_optim
    encd = SentenceEncoder(len(dictionary), VEC_SIZE).load(
        '.', from_version, device)
    clsf = Classifier(VEC_SIZE, dataset.CATAGORIES).load(
        '.', from_version, device)
    encd_optim = optim.SGD(encd.parameters(), lr=lr)
    clsf_optim = optim.Adam(clsf.parameters(), lr=lr, betas=betas)
    for i in range(epoches):
        train_epoch()
        encd.save('.', i+from_version)
        clsf.save('.', i+from_version)
    pass


def test_input():
    while True:
        sent = input('sent2vec> ')
        sent = dictionary[sent]
        sent = torch.LongTensor(sent)
        with torch.no_grad():
            result = encd(sent, hidden)
        result = np.array(result).reshape(8, 16)*255
        img = Image.fromarray(result.astype(np.uint8))
        img = img.resize((128, 64))
        img.show()


train()
