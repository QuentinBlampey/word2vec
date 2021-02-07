import re
from functools import reduce

from torch.utils.data import Dataset

class SentencesDataset(Dataset):
    """
    Very simple dataset composed of a list of sentences.
    """
    def __init__(self, data_filename, window_size=5):
        """
        window_size must be odd
        """
        super(SentencesDataset, self).__init__()
        with open(data_filename, "r") as f:
            self.sentences = [l for l in f]

        self.window_size = window_size
        self.preprocess()

        index, sentence_index = 0, 0
        self.mapping = []
        for i,s in enumerate(self.sentences):
            for j,_ in enumerate(s[2:-2],2):
                self.mapping.append((i,j))

    def tokenize(self, sentence):
        sentence = re.sub("\d+", "number_placeholder", sentence)
        return re.findall(r"\w+", sentence)

    def preprocess(self):
        self.sentences = list(map(self.tokenize, self.sentences))
        self.sentences = list(filter(lambda l: len(l)>=self.window_size, self.sentences))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self,index):
        i,j = self.mapping[index]
        s = self.sentences[i]
        neighbors = s[j-2:j] + s[j+1:j+3]
        return s[j], neighbors