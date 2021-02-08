from functools import reduce
import argparse
import json
from word2vec import Word2Vec
from dataset import WordPairsDataset, Loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_filename", type=str, default="example_dataset.txt")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-emb", "--embedding_size", type=int, default=10)
    parser.add_argument("-lr", "--lr", type=float, default=.1)
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")

    dataset = WordPairsDataset(args.data_filename)

    vocab = set(reduce(lambda x,y: x+y, dataset.sentences))
    print(f"> {len(vocab)} words\n")

    word2vec = Word2Vec(vocab, args.embedding_size, learning_rate=args.lr)
    loader = Loader(dataset, args.batch_size, word2vec.word_to_index)

    for epoch in range(args.epochs):
        losses = []
        for input_indices, output_indices in loader:
            losses.append(word2vec.step(input_indices, output_indices))
        print(f"Epoch {epoch:02} - Loss {sum(losses)/len(losses):.3f}")