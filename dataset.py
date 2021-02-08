import re
from math import ceil
import numpy as np

class WordPairsDataset:
    """
    Very simple dataset composed of pairs of words
    """
    def __init__(self, data_filename, window_size=5):
        """
        window_size must be odd
        """
        super(WordPairsDataset, self).__init__()
        self.pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*') # Tokens containing an alpha symbol
        self.padding = window_size // 2
        
        with open(data_filename, "r") as f:
            self.sentences = [l for l in f]

        self.window_size = window_size
        self.preprocess()
        
        self.pairs = []
        for sentence in self.sentences:
            for i, word in enumerate(sentence):
                for delta in range(-self.padding, self.padding+1):
                    if delta != 0 and i + delta >= 0 and i + delta < len(sentence):
                        self.pairs.append((word, sentence[i + delta]))
                    

    def tokenize(self, sentence):
        return self.pattern.findall(sentence.lower())

    def preprocess(self):
        self.sentences = list(map(self.tokenize, self.sentences))
        self.sentences = list(filter(lambda l: len(l) >= self.window_size, self.sentences))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

class Loader:
    def __init__(self, dataset, batch_size, word_to_index):
        self.dataset = dataset
        self.batch_size = batch_size
        self.word_to_index = word_to_index
        
        self.n_batches = ceil(len(self.dataset) / batch_size)
        self.current_index = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        input_indices, output_indices = [], []
        
        for input_word, output_word in self.dataset[self.current_index:self.current_index+self.batch_size]:
            input_indices.append(self.word_to_index[input_word])
            output_indices.append(self.word_to_index[output_word])
            
        self.current_index += self.batch_size
        if self.current_index < len(self.dataset):
            return np.array(input_indices), np.array(output_indices)
        else:
            self.current_index = 0
            raise StopIteration