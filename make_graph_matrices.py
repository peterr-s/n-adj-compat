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
sigmoid_cutoff = settings["sigmoid_cutoff"]
dropout_retention = settings["dropout_retention"]
n_mat_ct = settings["n_mat_ct"]
a_mat_ct = settings["a_mat_ct"]

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

x = tf.placeholder(tf.float32, shape = [2 * embedding_dim, batch_size], name = "x")
y = tf.placeholder(tf.float32, shape = [1, batch_size], name = "y");

# TODO fix other modules so that we don't need to split it here
# n adj
(x_n, x_a) = tf.split(x, 2, axis = 0)	# embedding_dim x batch_size

# transformation tensors
mat_ct = a_mat_ct + n_mat_ct
w_n = tf.get_variable("w_n", shape = [n_mat_ct, embedding_dim, embedding_dim])
w_a = tf.get_variable("w_a", shape = [a_mat_ct, embedding_dim, embedding_dim])
b_t = tf.get_variable("b_t", shape = [mat_ct, embedding_dim, batch_size])

transformed = tf.concat(                                # mat_ct x embedding_dim x batch_size
		[
                    tf.matmul(                          # a_mat_ct x embedding_dim x batch_size
                        w_a,                            # a_mat_ct x embedding_dim x embedding_dim
                        tf.tile(                        # a_mat_ct x embedding_dim x batch_size
                            tf.reshape(                 # 1 x embedding_dim x batch_size
                                x_n,
                                [1] + list(x_n.shape)
                                ),
                            [a_mat_ct, 1, 1]
                            )
                        ),
                    tf.matmul(                          # n_mat_ct x embedding_dim x batch_size
                        w_n,                            # n_mat_ct x embedding_dim x embedding_dim
                        tf.tile(                        # n_mat_ct x embedding_dim x batch_size
                            tf.reshape(                 # 1 x embedding_dim x batch_size
                                x_a,
                                [1] + list(x_a.shape)
                                ),
                            [n_mat_ct, 1, 1]
                            )
                        )
                ],
		axis = 0
		) + b_t


# confidence (output) layer
w_o1 = tf.get_variable("w_o1", shape = [mat_ct, 1, embedding_dim])
w_o2 = tf.get_variable("w_o2", shape = [1, a_mat_ct + n_mat_ct])
b_o1 = tf.get_variable("b_o1", shape = [mat_ct, batch_size])        # broadcast to mat_ct x batch_size
b_o2 = tf.get_variable("b_o2", shape = [1, batch_size])             # broadcast to batch_size

y_pred = cutoff_sigmoid(
		tf.matmul(
			w_o2,
			tf.nn.leaky_relu(
                            tf.squeeze(
				tf.matmul(
					w_o1,
					tf.nn.leaky_relu(transformed),
					)
				) + b_o1
                            )
			) + b_o2,
		name = "y_pred"
		)

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

