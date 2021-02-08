import numpy as np

def softmax(array):
    array = np.exp(array)
    return array / np.sum(array, axis=1, keepdims=True)

class Word2Vec():
    def __init__(self, vocab, embedding_size, learning_rate=.1):
        np.random.seed(0)

        self.vocab = vocab
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        
        self.vocab_size = len(self.vocab)
        self.word_to_index = { word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = { i: word for word, i in self.word_to_index.items()}
        
        self.embedding = np.random.randn(self.vocab_size, self.embedding_size)
        self.linear = np.random.randn(self.embedding_size, self.vocab_size)
        
    def one_hot_encod(self, output_indices):
        one_hot = np.zeros((len(output_indices), self.vocab_size))
        one_hot[np.arange(len(output_indices)), output_indices] = 1
        return one_hot
    
    def forward(self, input_indices):
        embeddings = self.embedding[input_indices, :]
        return embeddings, softmax(embeddings @ self.linear)
    
    def criterion(self, y, outputs):
        return -1/self.vocab_size*np.sum(y*np.log(outputs), axis=1).mean()
    
    def step(self, input_indices, output_indices):
        y = self.one_hot_encod(output_indices)
        embeddings, outputs = self.forward(input_indices)

        dL_dx = 1/self.vocab_size*(outputs-y)

        grad_linear = embeddings.T @ dL_dx
        self.linear -= self.learning_rate*grad_linear
        
        grad_embedding = dL_dx @ self.linear.T
        self.embedding[input_indices, :] -= self.learning_rate*grad_embedding

        return self.criterion(y, outputs)