#!/usr/bin/python3

import tensorflow as tf

# hyperparameters
batch_size = 1000
embedding_dim = 300
hidden_sizes = [10000]

# mitchell and lapata
# papers on embeddings, compatibility in parsing

#normalize batch
x = tf.placeholder(tf.float32, shape = [2 * embedding_dim, batch_size], name = "x")
mi = tf.placeholder(tf.float32, shape = [1, batch_size], name = "mi")
y = tf.placeholder(tf.float32, shape = [2, batch_size], name = "y");

# create and link hidden layers
hidden = x
for (i, hidden_size) in enumerate(hidden_sizes) :
    w = tf.get_variable("w_h_%i" % i, shape = [hidden_size, hidden.shape[0]])
    b = tf.get_variable("b_h_%i" % i, shape = [hidden_size, 1])

    hidden = tf.nn.relu(tf.matmul(w, hidden) + b) # leaky?
    hidden = tf.nn.dropout(hidden, 0.95) # dropout (5%? 10%?)

# confidence (output) layer
    w = tf.get_variable("w_o", shape = [y.shape[0], hidden.shape[0]])
    b = tf.get_variable("b_o", shape = [batch_size])
    y_pred = tf.sigmoid(tf.matmul(w, hidden) + b, name = "y_pred")

    loss = tf.losses.log_loss(y, y_pred, name = "loss")
    accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y, axis = 0), tf.argmax(y_pred, axis = 0)),
                tf.float32), # use float here so that the average is float
            name = "accuracy")
    train = tf.train.AdamOptimizer(0.015).minimize(loss, name = "train")
#normgd adagrad

# initialization op
init = tf.variables_initializer(tf.global_variables(), name = "init")

# save op
saver = tf.train.Saver(tf.global_variables())

# write graph to file
definition = tf.Session().graph_def
tf.train.write_graph(definition, ".", "graph.pb", as_text = False)

