import torch
import torch.nn as nn
from torch import optim
import torch.functional as F
from model import EmbeddingRNN, Classifier
from utils import WordDict, QADataset

lr = 0.01

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss()

rnn = EmbeddingRNN(30000).to(device)
clf = Classifier(64).to(device)

dataset = QADataset('train-seg.pth')
word_dict = WordDict('word-dict.txt')

rnn_optim = optim.SGD(rnn.parameters(), lr=lr)
clf_optim = optim.SGD(clf.parameters(), lr=lr)

REAL_LABEL = torch.ones(1).to(device)
FAKE_LABEL = torch.zeros(1).to(device)


def train_epoch():
    loss_t = 0
    item_c = 0

    for i, data in enumerate(dataset):
        question, reals, fakes = word_dict[data]

        real_loss = torch.zeros(1).to(device)
        fake_loss = torch.zeros(1).to(device)

        question = [word_dict.START] + question + [word_dict.END]
        question = torch.LongTensor(question).to(device)

        hidden = torch.zeros(1, 1, 64).to(device)

        for j in reals:
            rnn_optim.zero_grad()
            clf_optim.zero_grad()

            qust = rnn(question, hidden, end=True)

            j = [word_dict.START] + j + [word_dict.END]
            j = torch.LongTensor(j).to(device)
            answer = rnn(j, hidden, end=True)

            result = clf(qust, answer)

            loss = criterion(result.view(1), REAL_LABEL)
            loss_t += loss
            loss.backward()

            rnn_optim.step()
            clf_optim.step()

        for j in fakes:
            rnn_optim.zero_grad()
            clf_optim.zero_grad()

            qust = rnn(question, hidden, end=True)

            j = [word_dict.START] + j + [word_dict.END]
            j = torch.LongTensor(j).to(device)
            answer = rnn(j, hidden, end=True)

            result = clf(qust, answer)

            loss = criterion(result.view(1), FAKE_LABEL)
            loss_t += loss
            loss.backward()

            rnn_optim.step()
            clf_optim.step()
        
        '''
        if len(reals):
            real_loss /= len(reals)
        if len(fakes):
            fake_loss /= len(fakes)

        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        '''
        item_c += len(reals) + len(fakes)

        if i % 30 == 0:
            print(i, loss_t.item() / item_c)
            loss_t = 0
            item_c = 0


for i in range(5):
    train_epoch()
    torch.save(rnn.state_dict(), 'rnn.pth')
