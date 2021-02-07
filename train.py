from functools import reduce
from torch.utils.data import DataLoader

from word2vec import *
from dataset import *

data_filename = "example_dataset.txt"
batch_size = 1 # Must be 1 for now
n_epochs = 10

sd = SentencesDataset(data_filename)
loader =  DataLoader(sd, batch_size=batch_size, shuffle=True)

words = reduce(lambda x,y: x+y, sd.sentences)
vocab = set(words)
embedding_size = 10

word2vec = Word2Vec(vocab, embedding_size)

for epoch in range(n_epochs):
    loss = 0
    for word, neighbors in loader:
        word = word[0] # CRAPPY
        neighbors = list(map(lambda x:x[0], neighbors)) # CRAPPY
        loss += word2vec.step_from_words(word, neighbors)
    print(f"Loss - {loss:.3f}")