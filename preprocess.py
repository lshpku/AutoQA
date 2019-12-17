import pkuseg
import pickle
from utils import char_dict
import numpy as np

seg = pkuseg.pkuseg()  # load the default model


def train_seg(path: str) -> list:
    # Each is (question, [real1, ...], [fake1, ...])
    # Note: There do be some f**king questions that have
    #       no/multiple real answers or no fake answers.
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
                items.append((seg.cut(question), [], []))
                count += 1
                print('\rtokenizing: {}'.format(count), end='')
            content = seg.cut(line[1])
            if int(line[2]):  # real answer
                items[count-1][1].append(content)
            else:             # fake answer
                items[count-1][2].append(content)
        print()
    return items


def make_dict(items: list) -> list:
    # Count words
    word_dict = {}  # {word: number}
    for item in items:
        for sent in [item[0]]+item[1]+item[2]:
            for word in sent:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

    # Remove words that repeat too few
    word_list = [(j, i) for i, j in word_dict.items() if j >= 12]
    word_list.sort()
    word_list.reverse()
    return word_list


def seg_chars(path):
    with open(path, 'r') as f:
        text = f.read()
    items = []
    question = ''
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split('\t')
            if question != line[0]:  # new question
                question = line[0]
                items.append((char_dict[question], [], []))
                print('\r{}'.format(len(items)), end='')
            if not line[1]:
                continue
            content = char_dict[line[1]]
            if isinstance(content, int):  # BugFix: must be list
                content = [content]
            if int(line[2]):  # real answer
                items[-1][1].append(content)
            else:             # fake answer
                items[-1][2].append(content)
        print()
    with open('seg_chars.pth', 'wb') as f:
        pickle.dump(items, f)

seg_chars('data/validation-set.data')

'''
items = train_seg('data/validation-set.data')

with open('valid-seg.pth', 'wb') as f:
    pickle.dump(items, f)

with open('train-seg.pth', 'rb') as f:
    items = pickle.load(f)

word_list = make_dict(items)
print('word count: {}'.format(len(word_list)))

with open('word-dict.txt', 'w') as f:
    for i, j in word_list:
        f.write('{}\t{}\n'.format(i, j))
'''