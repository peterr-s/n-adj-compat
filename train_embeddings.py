#!/usr/bin/python3

import gensim.models
import sys
import json

# read settings file
settings_file = open("settings.json", "r")
settings = json.load(settings_file)
settings_file.close()

sentences = []

if len(sys.argv) > 1 :
    path = sys.argv[1]

    input = open(path);
    for line in input.read().split("\n") :
        sentences += [line.split()[1:]] # skip root
    input.close()
    del(input)
else :
    sentences += [["__", "__", "__"]]

model = gensim.models.Word2Vec(sentences, size = settings["embedding_dim"], window = 10, min_count = settings["min_embed_ct"])

# not really clear that binary True/False makes a difference, but the other module is forced to read as text so that's how we'll do it
model.wv.save_word2vec_format("./temp-embeddings", binary=False)

# cut off first line because the other module doesn't like it
temp_file = open("./temp-embeddings")
model_file = open(settings["embedding_file"], "w")
buf = ""
for line in temp_file.read().split("\n")[1:] :
    buf += line
    buf += "\n"
#    model_file.write(line)
#    model_file.write("\n")
buf = buf.strip()
model_file.write(buf)

model_file.close()
temp_file.close()

