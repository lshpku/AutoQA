#from torch.utils.data import Dataset, DataLoader
import pickle


class WordDict():
    def __init__(self, path: str):
        # Special tags
        self.START = 0
        self.END = 1
        self.UNKNOWN = 2

        # Load word dict
        with open(path, 'r') as f:
            word_list = f.read()
        word_list = word_list.split('\n')
        word_list = [i.split('\t')[1] for i in word_list if i]
        self.__dict = {i: j+3 for j, i in enumerate(word_list)}

    def __len__(self):
        return len(self.__dict) + 3

    def __getitem__(self, word):
        if isinstance(word, str):
            if word in self.__dict:
                return self.__dict[word]
            return self.UNKNOWN
        return [self[i] for i in word]  # recurse for list


class QADataset():
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            self.__items = pickle.load(f)

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, index):
        return self.__items[index]
