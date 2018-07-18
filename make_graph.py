#!/usr/bin/python3

import tensorflow as tf
import json

# read settings file
settings_file = open("settings.json", "r")
settings = json.load(settings_file)
settings_file.close()

# hyperparameters
batch_size = settings["batch_size"]
embedding_dim = settings["embedding_dim"]
hidden_sizes = settings["hidden_sizes"]
sigmoid_cutoff = settings["sigmoid_cutoff"]

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

# mitchell and lapata
# papers on embeddings, compatibility in parsing

#normalize batch
x = tf.placeholder(tf.float32, shape = [2 * embedding_dim, batch_size], name = "x")
y = tf.placeholder(tf.float32, shape = [1, batch_size], name = "y");

# create and link hidden layers
hidden = x
for (i, hidden_size) in enumerate(hidden_sizes) :
    w = tf.get_variable("w_h_%i" % i, shape = [hidden_size, hidden.shape[0]])
    b = tf.get_variable("b_h_%i" % i, shape = [hidden_size, 1])

    hidden = tf.nn.relu(tf.matmul(w, hidden) + b) # leaky?
    hidden = tf.nn.dropout(hidden, settings["dropout_retention"]) # dropout (5%? 10%?)

# confidence (output) layer
w = tf.get_variable("w_o", shape = [y.shape[0], hidden.shape[0]])
b = tf.get_variable("b_o", shape = [batch_size])
y_pred = cutoff_sigmoid(tf.matmul(w, hidden) + b, name = "y_pred")

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

