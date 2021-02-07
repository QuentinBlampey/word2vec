import numpy as np

def softmax(array):
    array = np.exp(array)
    return array / np.sum(array)

class Word2Vec():
    def __init__(self, vocab, embedding_size, learning_rate=1.):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        
        self.vocab_size = len(self.vocab)
        self.word_to_index = { word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = { i: word for word, i in self.word_to_index.items()}
        
        self.embedding = np.random.randn(self.vocab_size, self.embedding_size)
        self.linear = np.random.randn(self.embedding_size, self.vocab_size)
        
    def one_hot_encod(self, index):
        one_hot = np.zeros(self.vocab_size)
        one_hot[index] = 1
        return one_hot
    
    def forward(self, index):
        emb = self.embedding[index,:]
        return emb, softmax(emb @ self.linear)
    
    def criterion(self, y, output):
        return -1/self.vocab_size*np.inner(y, np.log(output))
    
    def step(self, input_index, output_index):
        y = self.one_hot_encod(output_index)
        emb, output = self.forward(input_index)

        loss = self.criterion(y, output)
        print(f"Loss - {loss:.3f}")

        dL_dx = 1/self.vocab_size*(output-y)

        grad_linear = np.outer(emb, dL_dx)
        self.linear -= self.learning_rate*grad_linear

        grad_embedding = self.linear @ dL_dx
        self.embedding[input_index,:] -= self.learning_rate*grad_embedding