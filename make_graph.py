#!/usr/bin/python3

import tensorflow as tf

# hyperparameters
batch_size = 10
embedding_dim = 300
hidden_sizes = [1000]

x = tf.placeholder(tf.float32, shape = [2 * embedding_dim, batch_size], name = "x")
y = tf.placeholder(tf.float32, shape = [1, batch_size], name = "y")

# create and link hidden layers
hidden = x
for (i, hidden_size) in enumerate(hidden_sizes) :
    w = tf.get_variable("w_h_%i" % i, shape = [hidden_size, hidden.shape[0]])
    b = tf.get_variable("b_h_%i" % i, shape = [hidden_size, 1])

    hidden = tf.sigmoid(tf.matmul(w, hidden) + b)

# output layer
w = tf.get_variable("w_o", shape = [y.shape[0], hidden.shape[0]])
b = tf.get_variable("b_o", shape = [batch_size])
y_pred = tf.matmul(w, hidden, name = "y_pred") + b

# phases are managed in the main module
loss = tf.reduce_mean(tf.square(y_pred - y), name = "loss")
train = tf.train.AdamOptimizer(0.005).minimize(loss, name = "train")

init = tf.variables_initializer(tf.global_variables(), name = "init")

# write graph to file
definition = tf.Session().graph_def
tf.train.write_graph(definition, ".", "model.pb", as_text = False)

