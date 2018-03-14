
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from tagger.data_utils import Progbar, minibatches, pad_sequences, plot_confusion_matrix
# from tagger.data_helper import

from utils import constants


class SRLModel(object):
    def __init__(self, config, embeddings, n_tags, nchars=None):
        """
        Args:
            config: class with hyper parameters
            embeddings: np array with embeddings
            nchars: (int) size of chars vocabulary
        """
        self.config = config
        self.embeddings = embeddings
        self.nchars = nchars
        self.ntags = n_tags


    def add_placeholders(self):
        """
        Adds placeholders to self
        """

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of words.
                A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if self.config.chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, self.config.label_dict.get_index(constants.unknown_label))
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
                                                     name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.chars:
                # get embeddings matrix
                _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                                                   shape=[self.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
                                                         name="char_embeddings")
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])

                # bi lstm on chars
                # need 2 instances of cells since tf 1.1
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, tate_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                      cell_bw, char_embeddings,
                                                                                      sequence_length=word_lengths,
                                                                                      dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], 2 * self.config.char_hidden_size])

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("model"):

            # Build model for a bi-directional recurrent architectures
            if self.config.model_type == "bi-lstm":

                with tf.variable_scope("bi-lstm"):
                    # cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size)
                    # cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size)

                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)

                    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                cell_bw, self.word_embeddings,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
                    output = tf.concat([output_fw, output_bw], axis=-1)
                    output = tf.nn.dropout(output, self.dropout)

                with tf.variable_scope("projection"):
                    W = tf.get_variable("W", shape=[2 * self.config.hidden_size, self.ntags],
                                        dtype=tf.float32)

                    b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

                    ntime_steps = tf.shape(output)[1]
                    output = tf.reshape(output, [-1, 2 * self.config.hidden_size])
                    pred = tf.matmul(output, W) + b
                    self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

                    return

            # Build model for all other one-directional recurrent architectures
            else:
                if self.config.model_type == "rnn":
                    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicRNNCell
                    cell = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size)

                elif self.config.model_type == "lstm":
                    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
                    cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)

                elif self.config.model_type == "gru":
                    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/GRUCell
                    cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)

                else:
                    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell
                    cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)


                # multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell], 3)
                # outputs, state = tf.nn.dynamic_rnn(multi_layer_cell, self.word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, self.word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)

                output = tf.nn.dropout(outputs, self.dropout)


                with tf.variable_scope("projection"):
                    W = tf.get_variable("W", shape=[self.config.hidden_size, self.ntags],
                                        dtype=tf.float32)

                    b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

                    ntime_steps = tf.shape(output)[1]
                    output = tf.reshape(output, [-1, self.config.hidden_size])

                    pred = tf.matmul(output, W) + b
                    self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])


    def add_pred_op(self):
        """
        Adds labels_pred to self
        """
        if not self.config.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """
        Adds loss to self
        """
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # For TensorBoard
        tf.summary.scalar("loss", self.loss)

    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            # Stochastic Gradient Descent Method
            if self.config.learning_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.learning_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.config.learning_method == 'sgd':
                # Vanilla Gradient Descent
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif self.config.learning_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Unknown train op {}".format(
                    self.config.lr_method))

            # gradient clipping if config.clip is positive
            if self.config.clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        # Visualize results in TensorBoard
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.model_output, sess.graph)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

        tf.summary.scalar("learning_rate", self.lr)
        # tf.summary.scalar("logits", self.logits)
        # tf.summary.scalar("transition_params", self.transition_params)
        # tf.summary.scalar("train_op", self.train_op)

    def predict_batch(self, sess, words):
        """
        Args:
            sess: a tensorflow session
            words: list of sentences
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """
        # get the feed dictionnary
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, transition_params)
                viterbi_sequences += [viterbi_sequence]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, sess, x_train, x_dev, y_train, y_dev, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
            epoch: (int) number of the epoch
        """

        nbatches = (len(x_train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)

        for i, (words, labels) in enumerate(minibatches(x_train, y_train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            # prog.update(i + 1, [("train loss", train_loss)])

            # TensorBoard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        # printer.print_to_console("Testing model over Development Set:")
        acc, precision, recall, f1 = self.run_evaluate(sess, x_dev, y_dev)

        return acc, f1, train_loss


    def run_evaluate(self, sess, x_dev, y_dev, test_mode=False):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """

        # Measures to calculate Accuracy, Precision, Recall and F1-Score
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0

        y_true = []
        y_pred = []
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for words, labels in minibatches(x_dev, y_dev, self.config.batch_size):

            labels_pred, sequence_lengths = self.predict_batch(sess, words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):

                y_true = np.concatenate((y_true, lab), axis=0)
                y_pred = np.concatenate((y_pred, lab_pred), axis=0)


        # Compute confusion matrix
        matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)

        if test_mode:
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(matrix, classes=self.config.label_dict.str2idx.items(),
                                  title='Confusion matrix, without normalization')

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(matrix, classes=self.config.label_dict.str2idx.items(), normalize=True,
                                  title='Normalized confusi443ss3on matrix')

            plt.show()

        # Count the number of examples in the evaluation set
        n_examples = len(x_dev)

        # Check which evaluation method is defined
        # 'Word':   Calculate results based on single tokens
        # 'Entity': Calculate results based on entities (spans of tokens, e.g., "B_where" "I_where"
        if self.config.eval_level == "word":
            for words, labels in minibatches(x_dev, y_dev, self.config.batch_size):

                labels_pred, sequence_lengths = self.predict_batch(sess, words)

                for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]

                    accuracy += metrics.accuracy_score(lab, lab_pred, normalize=True)
                    precision += metrics.precision_score(lab, lab_pred, average=self.config.score_method)
                    recall += metrics.recall_score(lab, lab_pred, average=self.config.score_method)
                    f1 += metrics.f1_score(lab, lab_pred, average=self.config.score_method)

        else:
            for words, labels in minibatches(x_dev, y_dev, self.config.batch_size):

                labels_pred, sequence_lengths = self.predict_batch(sess, words)

                for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]

                    lab_results = []
                    pred_results = []
                    old_label = ""

                    for g, p in zip(lab, lab_pred):

                        if g == "":
                            lab_results.append([g])
                            pred_results.append([p])
                            old_label = g

                        else:
                            if g == self.config.label_dict.get_index("b_where"):
                                lab_results.append([g])
                                pred_results.append([p])
                                old_label = g
                            elif old_label == g:
                                lab_results[len(lab_results) - 1].append(g)
                                pred_results[len(lab_results) - 1].append(p)
                                old_label = g
                            elif g == self.config.label_dict.get_index("i_where"):
                                lab_results[len(lab_results) - 1].append(g)
                                pred_results[len(lab_results) - 1].append(p)
                                old_label = g
                            else:
                                lab_results.append([g])
                                pred_results.append([p])
                                old_label = g

                    gold = []
                    pred = []

                    for a, b in zip(lab_results, pred_results):

                        lab_text = ""
                        pred_text = ""

                        if len(a) > 1:
                            for a_item, b_item in zip(a, b):
                                lab_text += str(a_item)
                                pred_text += str(b_item)
                        else:
                            lab_text = str(a[0])
                            pred_text = str(b[0])

                        gold.append(lab_text)
                        pred.append(pred_text)

                    accuracy += metrics.accuracy_score(gold, pred, normalize=True)
                    precision += metrics.precision_score(gold, pred, average=self.config.score_method)
                    recall += metrics.recall_score(gold, pred, average=self.config.score_method)
                    f1 += metrics.f1_score(gold, pred, average=self.config.score_method)


        results = [["%.2f" % (accuracy / n_examples), "%.2f" % (recall / n_examples), "%.2f" % (precision / n_examples), "%.2f" % (f1 / n_examples)]]
        result_df = pd.DataFrame(results, columns=["Accuracy", "Recall", "Precision", "F1"])

        # printer.print_to_console(result_df)
        # printer.print_to_console("")

        return (accuracy / n_examples), (precision / n_examples), (recall / n_examples), (f1 / n_examples)


    def train(self, x_train, x_dev, y_train, y_dev):
        """
        Performs training with early stopping and learning rate exponential decay

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
        """

        acc = 0
        f1 = 0
        loss = 0
        best_score = 0

        # saver = tf.train.Saver('my-model')
        saver = tf.train.Saver()

        # for early stopping
        n_epoch_no_imprv = 0
        n_epoch_stagnating = 0
        previous_score = 0

        with tf.Session() as sess:
            sess.run(self.init)
            if self.config.reload:
                saver.restore(sess, self.config.model_output)

            # TensorBoard
            self.add_summary(sess)
            for epoch in range(self.config.n_epochs):
                # printer.print_to_console("")
                # printer.print_to_console("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))

                acc, f1, loss = self.run_epoch(sess, x_train, x_dev, y_train, y_dev, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    n_epoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1

                # Check whether the model is improving its results
                elif f1 == previous_score:
                    n_epoch_stagnating += 1
                    if n_epoch_stagnating >= self.config.n_epoch_stagnating:
                        print("Abort training due to 5 epochs without learning!")
                        # sess.close()

                        return acc, f1, loss

                        # break

                # Check whether the model is learning or not
                else:
                    n_epoch_no_imprv += 1
                    if n_epoch_no_imprv >= self.config.n_epoch_no_imprv:
                        # printer.print_to_console("Early stopping {} epochs without improvement".format(n_epoch_no_imprv))
                        # printer.print_to_console("")
                        # sess.close()

                        return acc, f1, loss

                        # break

                # Save the current result for the next iteration/epoch
                previous_score = f1

            # Return last results after n_epochs
            return acc, f1, loss


    def evaluate(self, x_test, y_test, test_mode=False):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)

            # printer.print_to_console("Testing model over Test Set:")
            accuracy, precision, recall, f1 = self.run_evaluate(sess, x_test, y_test, test_mode=test_mode)

            # sess.close()

        return accuracy, precision, recall, f1
