#!/usr/bin/python3

import tensorflow as tf
import json
from gensim.models import KeyedVectors

# read settings file
settings_file = open("settings.json", "r")
settings = json.load(settings_file)
settings_file.close()

# hyperparameters
batch_size = settings["batch_size"]
embedding_dim = settings["embedding_dim"]
hidden_sizes = settings["hidden_sizes"]
sigmoid_cutoff = settings["sigmoid_cutoff"]
mat_width = settings["mat_width"]

# this prevents the weights from being perpetually adjusted after they're good enough
def cutoff_sigmoid(x, name = None) :
    pos_mask = tf.where(
            tf.greater(x, sigmoid_cutoff),
            x = tf.fill(x.shape, 1.0),
            y = tf.sigmoid(x)
            )
    neg_mask = tf.where(
            tf.less(pos_mask, -sigmoid_cutoff),
            x = tf.fill(x.shape, 0.0),
            y = pos_mask,
            name = name
            )
    return neg_mask

# for numbering embeddings so they can be used as keys in embedding_lookup
class Numberer :
    def __init__(self) :
        self.eton = dict()
        self.ntoe = list()

    def number(self, embedding) :
        n = self.eton.get(embedding)

        if n is None :
            n = len(self.ntoe)
            self.eton[embedding] = n
            self.ntoe.append(embedding)

        return n

    def value(self, number) :
        return tf.gather(self.ntoe, number)

n_numberer = Numberer()
a_numberer = Numberer()

# mitchell and lapata
# papers on embeddings, compatibility in parsing

# get model purely so that we know how many matrices are needed
model = KeyedVectors.load_word2vec_format(settings["embedding_file"], binary = True)
word_ct = len(model.wv.vocab)

# train different matrices in case words occur as both n and adj
n_matrices = tf.get_variable("n_matrices", shape = [word_ct, mat_width, embedding_dim])
a_matrices = tf.get_variable("a_matrices", shape = [word_ct, mat_width, embedding_dim])

x = tf.placeholder(tf.float32, shape = [2 * embedding_dim, batch_size], name = "x")
y = tf.placeholder(tf.float32, shape = [1, batch_size], name = "y");

# TODO fix other modules so that we don't need to split it here
# n adj
(x_n, x_a) = tf.split(x, 2, axis = 0)

# get matrices for each
# transpose matrix so that map_fn is applied to each element in the batch
# map using numberer
# this converts from rank 3 float to rank 2 int
n_mat = tf.nn.embedding_lookup(
            n_matrices,
            tf.map_fn(
                n_numberer.number,
                tf.transpose(x_n),
                dtype = tf.int32
                )
            )
a_mat = tf.nn.embedding_lookup(
            a_matrices,
            tf.map_fn(
                a_numberer.number,
                tf.transpose(x_a),
                dtype = tf.int32
                )
            )

# multiply by matrices of counterparts
# transpose embeddings so that matrix multiplication is done elementwise
# add a dummy dimension
# a is batch_size x embedding_dim x embedding_dim
# b is batch_size x embedding_dim x 1
# prod batch_size x embedding_dim x 1
n_comp = tf.matmul(
            a_mat,
            tf.reshape(
                tf.transpose(x_n),
                [batch_size, embedding_dim, 1]
                )
            )
a_comp = tf.matmul(
            n_mat,
            tf.reshape(
                tf.transpose(x_a),
                [batch_size, embedding_dim, 1]
                )
            )

# combine to get confidence
# use the embedding axis to concatenate
# reduce null axis of x for dot prod with w; reduce to single number
# w is 1 x 2*embedding_dim
# x is 2*embedding_dim x batch_size
# b is batch_size
comp = tf.transpose(
        tf.squeeze(
            tf.concat(
                [n_comp, a_comp],
                axis = 1
                )
            )
        )
w = tf.get_variable("w", shape = [1, 2 * embedding_dim])
b = tf.get_variable("b", shape = [batch_size])
y_pred = cutoff_sigmoid(tf.matmul(w, comp) + b, name = "y_pred")

loss = tf.losses.log_loss(y, y_pred)
loss = tf.identity(loss, name = "loss")
accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.greater(y, 0.5), tf.greater(y_pred, 0.5)), # broadcasting allows us to compare constants elementwise
            tf.float32), # use float here so that the average is float
        name = "accuracy")
train = tf.train.AdamOptimizer(settings["learning_rate"]).minimize(loss, name = "train")
#normgd adagrad

# initialization op
init = tf.variables_initializer(tf.global_variables(), name = "init")

# save op
saver = tf.train.Saver(tf.global_variables())

# write graph to file
definition = tf.Session().graph_def
tf.train.write_graph(definition, ".", settings["graph_file"], as_text = False)

