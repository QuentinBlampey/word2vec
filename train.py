from functools import reduce

from word2vec import Word2Vec
from dataset import WordPairsDataset, Loader

data_filename = "example_dataset.txt"
batch_size = 4
n_epochs = 10

dataset = WordPairsDataset(data_filename)

words = reduce(lambda x,y: x+y, dataset.sentences)
vocab = set(words)
embedding_size = 10

word2vec = Word2Vec(vocab, embedding_size)
loader = Loader(dataset, batch_size, word2vec.word_to_index)

for epoch in range(n_epochs):
    loss = 0
    for input_indices, output_indices in loader:
        loss += word2vec.step(input_indices, output_indices)
    print(f"Loss - {loss:.3f}")