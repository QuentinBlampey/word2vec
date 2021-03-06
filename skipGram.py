from __future__ import division
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

from word2vec import Word2Vec
from dataset import WordPairsDataset, Loader

__authors__ = ['clement_piat','quentin_blampey']
__emails__  = ['clement.piat@student.ecp.fr', 'quentin.blampey@student.ecp.fr']

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')
	parser.add_argument('--not_xavier_uniform', action='store_true')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument("-e", "--epochs", type=int, default=1)
	parser.add_argument("-emb", "--embedding_size", type=int, default=10)
	parser.add_argument("-lr", "--lr", type=float, default=.1)

	opts = parser.parse_args()

	if not opts.test:
		print(vars(opts))

		dataset = WordPairsDataset(opts.text)

		counter = Counter(word for sentence in dataset.sentences for word in sentence)
		vocab = list(counter.keys())

		print(f"> {len(vocab)} words")
		print(f"> {len(dataset.pairs)} pairs\n")


		word2vec = Word2Vec(vocab, opts.embedding_size, learning_rate=opts.lr, xavier_uniform=not opts.not_xavier_uniform)

		probabilities = np.array([counter[word2vec.index_to_word[i]] for i in range(word2vec.vocab_size)])**(3/4)
		loader = Loader(dataset, opts.batch_size, word2vec.word_to_index, probabilities)

		for epoch in range(opts.epochs):
			losses = []
			for input_indices, output_indices, indices, word_indices in tqdm(loader):
				losses.append(word2vec.step(input_indices, output_indices, indices, word_indices))
			# print(losses)
			print(f"Epoch {epoch:02} - Loss {sum(losses)/len(losses):.3f}\n")

		word2vec.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		word2vec = Word2Vec.load(opts.model)
		for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(word2vec.similarity(a,b))