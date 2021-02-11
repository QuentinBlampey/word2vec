import numpy as np
import json

def softmax(array):
    array = np.exp(array)
    return array / np.sum(array, axis=1, keepdims=True)

class Word2Vec():
    def __init__(self, vocab, embedding_size=100, learning_rate=.1):
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
    
    def mask(self, indices, word_indices):
        mask = np.zeros((indices[-1] + 1, self.vocab_size))
        mask[indices, word_indices] = 1
        return mask

    def forward(self, input_indices):
        embeddings = self.embedding[input_indices, :]
        return embeddings, softmax(embeddings @ self.linear)
    
    def criterion(self, y, outputs):
        return -1/self.vocab_size*np.sum(y*np.log(outputs), axis=1).mean()
    
    def step(self, input_indices, output_indices, indices, word_indices):
        y = self.one_hot_encod(output_indices)
        embeddings, outputs = self.forward(input_indices)

        dL_dx = 1/self.vocab_size*(outputs-y)
        dL_dx = dL_dx*self.mask(indices, word_indices)

        grad_linear = embeddings.T @ dL_dx
        self.linear -= self.learning_rate*grad_linear
        
        grad_embedding = dL_dx @ self.linear.T
        self.embedding[input_indices, :] -= self.learning_rate*grad_embedding

        return self.criterion(y, outputs)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({"embedding": self.embedding.tolist(),
                       "word_to_index": self.word_to_index}, f, indent=4)

    @staticmethod
    def load(path):
        with open(path) as f:
            obj = json.load(f)
            word2vec = Word2Vec([], 0, 0)
            word2vec.embedding = np.array(obj['embedding'])
            word2vec.word_to_index = obj['word_to_index']
        return word2vec

    def similarity(self, word1, word2):
        try:
            emb1 = self.embedding[self.word_to_index[word1]]
            emb2 = self.embedding[self.word_to_index[word2]]
            return (1+np.inner(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))/2
        except:
            return 0.5