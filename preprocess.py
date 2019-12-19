import jieba
import pickle
import torch
from utils import WordDict
import numpy as np

MAXLEN_WORD = 4  # max length of a single word
#wd = WordDict('word-list.pkl')


def seg_sent(sent: str) -> list:
    '''
    Sentence -> [word1, word2, ...].
    '''
    words = list(jieba.cut(sent))
    sent = []
    for i in words:
        w = wd[i]
        if w == wd.UNK:  # split unknown words further
            sent += [wd[j] for j in i]
        else:
            sent.append(w)
    return sent


def parse_with_tag(path: str) -> list:
    '''
    Parse dataset with tags.\n
    Each item is a tuple (question, [real1, ...], [fake1, ...]).\n
    Note: There do be some f**king questions that have no/multiple
        real answers or no fake answers.
    '''
    items = []
    count = 0
    question = ''
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split('\t')
            if question != line[0]:  # new question
                question = line[0]
                items.append((seg_sent(question), [], []))
                count += 1
                print('\r{}'.format(count), end='')
            content = seg_sent(line[1])
            if not content:  # there do be empty sentences
                continue
            if int(line[2]):  # real answer
                items[count-1][1].append(content)
            else:             # fake answer
                items[count-1][2].append(content)
        print()
    with open('dataset-words.pkl', 'wb') as f:
        pickle.dump(items, f)


def parse_pretrained_weight(path: str):
    '''
    Extract words and vectors from pretrained weights.
    '''
    embd = open(path, 'r', encoding='utf-8')
    info = embd.readline().split()
    dict_size, vec_size = int(info[0]), int(info[1])
    print('dict_size:', dict_size)
    print('vec_size:', vec_size)

    word_dict = {}
    word_vec = []
    stats = [0] * 100

    while True:
        line = embd.readline()
        if not line:
            break
        line = line.split()
        if len(line[0]) < 100:
            stats[len(line[0])] += 1
        if len(line[0]) > MAXLEN_WORD:  # discard too long words
            continue
        if line[0] in word_dict:  # ignore repeating words
            continue
        word_dict[line[0]] = len(word_dict)
        word_vec.append([float(j) for j in line[1:]])
        print('\r{}'.format(len(word_dict)), end='')

    assert len(word_dict) == len(word_vec)
    print('\nreduced_size:', len(word_dict))
    embd.close()

    word_list = [None] * len(word_dict)
    for i, j in word_dict.items():
        word_list[j] = i
    with open('word-list.pkl', 'wb') as f:
        pickle.dump(word_list, f)
    with open('word-vec.pkl', 'wb') as f:
        pickle.dump(word_vec, f)
    print(stats)


parse_pretrained_weight('data/sgns.zhihu.char')
#parse_with_tag('data/train-set.data')
#parse_with_tag('data/validation-set.data')
