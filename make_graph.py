#!/usr/bin/python3

import tensorflow as tf

#class Phase(Enum) :
#    Train = 0
#    Validate = 1
#    Predict = 2

batch_size = 100
embedding_dim = 300
hidden_sizes = [1000]

x = tf.placeholder(tf.float32, shape = [batch_size, 2 * embedding_dim], name = "x")
y = tf.placeholder(tf.float32, name = "y", shape = [batch_size, 1])

hidden = x
for (i, hidden_size) in enumerate(hidden_sizes) :
    w = tf.get_variable("w_h_%i" % i, shape = [hidden.shape[1], hidden_size])
    b = tf.get_variable("b_h_%i" % i, shape = [hidden_size])

    hidden_out = tf.sigmoid(tf.matmul(hidden, w) + b)

    hidden = hidden_out

w = tf.get_variable("w_o", shape = [hidden.shape[1], 1])
b = tf.get_variable("b_o", shape = [1])
y_pred = tf.matmul(hidden, w) + b

loss = tf.reduce_mean(tf.square(y_pred - y))
train = tf.train.AdamOptimizer(0.005).minimize(loss, name = "train")

init = tf.variables_initializer(tf.global_variables(), name = "init")

definition = tf.Session().graph_def
tf.train.write_graph(definition, ".", "model.pb", as_text = False)

