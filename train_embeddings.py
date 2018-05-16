#!/usr/bin/python3

import gensim.models
import sys

path = "./text"
if len(sys.argv) > 1 :
    path = sys.argv[1]

input = open(path);
sentences = []
for line in input.read().split("\n") :
    sentences += [line.split()[1:]] # skip root
input.close()
del(input)

model = gensim.models.Word2Vec(sentences, size = 300, window = 10, min_count = 1) # make min count 50 for final test

# not really clear that binary True/False makes a difference, but the other module is forced to read as text so that's how we'll do it
model.wv.save_word2vec_format("./temp-embeddings" binary=False)

# cut off first line because the other module doesn't like it
temp_file = open("./temp-embeddings")
model_file = open("./embeddings", "w")
for line in model_file.read().split("\n")[1:] :
    model_file.write(line)
    model_file.write("\n")
model_file.close()
temp_file.close()

