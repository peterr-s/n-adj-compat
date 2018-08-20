#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import json
from gensim.models import KeyedVectors
from sklearn.cluster import MiniBatchKMeans

# read settings file
settings_file = open("settings.json", "r")
settings = json.load(settings_file)
settings_file.close()

# hyperparameters
batch_size = settings["batch_size"]
embedding_dim = settings["embedding_dim"]
sigmoid_cutoff = settings["sigmoid_cutoff"]
n_mat_ct = settings["n_mat_ct"]
a_mat_ct = settings["a_mat_ct"]
n_embed_retention = settings["n_embed_retention"]
a_embed_retention = settings["a_embed_retention"]

# make clusters for matrices
embedding_model = KeyedVectors.load_word2vec_format(settings["embedding_file"], binary = True)
n_clusterer = MiniBatchKMeans(n_clusters = n_mat_ct)
a_clusterer = MiniBatchKMeans(n_clusters = a_mat_ct)
n_set = set()
a_set = set()
for sample_file in settings["sample_files"] :
	pair_list = open(sample_file["path"])
	for line in pair_list.read().split("\n") :
		fields = line.split()
		if len(fields) == 2 :
			n_set.add(fields[0])
			a_set.add(fields[1])
n_embed_mat = np.array([embedding_model[e] for e in n_set if e in embedding_model])
a_embed_mat = np.array([embedding_model[e] for e in a_set if e in embedding_model])
n_clusterer = tf.constant(n_clusterer.fit_predict(n_embed_mat), dtype = tf.int32) # uint32 is not allowed as a tf idx type for some reason
a_clusterer = tf.constant(a_clusterer.fit_predict(a_embed_mat), dtype = tf.int32)
n_embed_mat = tf.constant(n_embed_mat)
a_embed_mat = tf.constant(a_embed_mat)

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

# train different matrices in case words occur as both n and adj
n_matrices = tf.get_variable("n_matrices", shape = [n_mat_ct, embedding_dim, embedding_dim])
a_matrices = tf.get_variable("a_matrices", shape = [a_mat_ct, embedding_dim, embedding_dim])

x = tf.placeholder(tf.float32, shape = [2 * embedding_dim, batch_size], name = "x")
y = tf.placeholder(tf.float32, shape = [1, batch_size], name = "y");

# TODO fix other modules so that we don't need to split it here
# n adj
(x_n, x_a) = tf.split(x, 2, axis = 0)

# get matrices for each
# get square of L2 norm of batch elementwise
# compare to dot prod of embedding model matrix with batch, matches are definite because everything has the same length (cf. def of cosine)
# get indices of matched elements on axis 0
# get centroid ids
# get transformation matrices
n_mat_idx = tf.gather(
                n_clusterer,                                # (embed_ct)
                tf.argmax(                                  # batch_size                    indices in embed_mat of matches
                    tf.cast(                                # (embed_ct) x batch_size
                        tf.equal(                           # (embed_ct) x batch_size       matches of sq L2 and dot prod
                            tf.matmul(                      # (embed_ct) x batch_size       dot prods of all X with all known embed
                                n_embed_mat,                # (embed_ct) x embedding_dim
                                x_n                         # embedding_dim x batch_size
                                ),
                            tf.reshape(                     # 1 x batch_size                sq L2 norm of each embedding
                                tf.reduce_sum(              # batch_size
                                    x_n * x_n,              # embedding_dim x batch_size
                                    axis = 0
                                    ),
                                shape = [1, batch_size]
                                )
                            ),
                        dtype = tf.uint8
                        ),
                    axis = 0
                    ),
                name = "n_mat_idx"
                )
a_mat_idx = tf.gather(
                a_clusterer,
                tf.argmax(
                    tf.cast(
                        tf.equal(
                            tf.matmul(a_embed_mat, x_a),
                            tf.reshape(
                                tf.reduce_sum(
                                    x_a * x_a,
                                    axis = 0
                                    ),
                                shape = [1, batch_size]
                                )
                            ),
                        dtype = tf.uint8
                        ),
                    axis = 0
                    ),
                name = "a_mat_idx"
                )
tf.cast(n_mat_idx, dtype = tf.float32, name = "n_mat_rs")
tf.cast(a_mat_idx, dtype = tf.float32, name = "a_mat_rs")
n_mat = tf.nn.embedding_lookup(
            n_matrices,
            n_mat_idx
            )
a_mat = tf.nn.embedding_lookup(
            a_matrices,
            a_mat_idx
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

