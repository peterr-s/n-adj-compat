#!/usr/bin/python3

import gensim.models
import sys

path = "./text"

if len(sys.argv) > 1 :
    path = sys.argv[1]

input = open(path);

sentences = []
for line in input.read().split("\n") :
    sentences += line

model = gensim.models.Word2Vec(sentences, size = 300, window = 10)
model.wv.save_word2vec_format("./embeddings")

