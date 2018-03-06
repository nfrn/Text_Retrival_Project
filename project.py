import os

import tensorflow as tf
from nltk import word_tokenize
import math




def load_data():
    tokens = list();
    words = list();
    with open("train.txt", 'r') as document:
        sentences = document.read().splitlines()
        for sentence in sentences:
            tokens.append(word_tokenize(sentence))

    for sentence in tokens:
        for word in sentence:
            words.append(word.lower())

    vocab = sorted(set(words))
    vocabulary_size=len(vocab);

    print(tokens)
    print(vocabulary_size)
    print(vocab)

def main():
    load_data()


if __name__ == '__main__':
    main()
