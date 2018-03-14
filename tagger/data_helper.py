# ======================================================================================================================
# Copyright © 2017 David Ganshorn. All rights reserved.
# Created by David Ganshorn on 08/22/2017.
#
# Last Updated by David Ganshorn on 08/22/2017.
#
# ======================================================================================================================

import os
import copy
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split as dev_split

from tagger.model import SRLModel
from tagger.data_utils import Progbar

from utils import constants, printer


def train_test_split(questions, config):

    # Placeholder for returned dataset
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for question in questions:

        # Temporary row
        token_row = []
        label_row = []

        for token, label in zip(question.tokens, question.semantic_labels):
            # Add token_id to the data
            token_row.append(config.word_dict.get_index(token))
            label_row.append(config.label_dict.get_index(label))

        if constants.tag_test_set.__contains__(int(question.id)):
            # Add new row to the data
            x_test.append(token_row)
            y_test.append(label_row)

        else:
            # Add new row to the data
            x_train.append(token_row)
            y_train.append(label_row)


    # Transform x to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test



# The k-fold cross- validation algorithm.
# It can be used to estimate generalization error of a learning algorithm A when the given dataset D is too small for a
# simple train / test or train / valid split to yield accurate estimation of generalization error, because the mean of a
# loss L on a small test set may have too high a variance.
def kFoldCrossValidation(x, y, model, optimizer, embeddings, config):

    # Array to store the best result and the corresponding combination
    best_config = []
    best_f1 = 0

    """ Hyper Parameters """
    # Train model with varying model hidden sizes
    hidden_size = [300, 200, 100, 50]
    # hidden_size = [100]

    # Varying learning rate number for different training results
    learning_rates = [0.1, 0.05, 0.01]
    # learning_rates = [0.001]

    # Varying learning rate decay for different training results
    learning_rate_decay = [1.0, 0.99, 0.95]
    # learning_rate_decay = [1.0]

    # Varying dropout probability different training results
    dropout_probabilities = [1.0, 0.75, 0.5]
    # dropout_probabilities = [1.0]

    # Define the number of steps to test all possibilities
    total_steps = len(hidden_size) * len(learning_rates) * len(learning_rate_decay) * len(dropout_probabilities)
    current_step = 0

    # Progressbar for better visualization
    prog = Progbar(target=total_steps)


    """ Test all Models with a different hidden size """
    for size in hidden_size:

        # Set global hidden size
        config.hidden_size = size

        """ Test all Models with different learning rate decays """
        for decay in learning_rate_decay:

            # Set Learning Rate Decay
            config.lr_decay = decay

            """ Test all Models with different learning rates """
            for rate in learning_rates:

                # Set dynamically the learning rate
                config.lr = rate

                """ Test all Models with different dropout probabilities """
                for prob in dropout_probabilities:
                    # Set dynamically the learning rate
                    config.dropout = prob

                    # Increase step and start timer
                    current_step += 1
                    start_time = time.time()

                    # Start Training for this combination
                    printer.print_to_file("Combination {} out of {}:".format(current_step, total_steps))
                    printer.print_to_file(
                        "===========================================================================================================")

                    accuracy, f1_score = fit_and_score(x, y, embeddings, config)

                    # Stop timer
                    end_time = time.time() - start_time

                    # Update Progressbar with new fold step
                    prog.update(current_step)

                    setting = [
                        [model, optimizer, size, rate, config.lr_decay, prob, "%.4f" % accuracy, "%.4f" % f1_score,
                         str("%.2f" % (current_step * 100 / total_steps)) + "%",
                         time.strftime("%H:%M:%S min.", time.gmtime(end_time))]]

                    # Convert global result list into Pandas Matrix for better visualization
                    setting_df = pd.DataFrame(setting,
                                              columns=["Model", "Optimizer", "Hidden", "Learning", "LR Decay",
                                                       "Dropout", "Accuracy", "F1", "Progress", "Duration"])

                    printer.print_to_file(setting_df)
                    printer.print_to_file("")
                    printer.print_to_file(
                        "-----------------------------------------------------------------------------------------------------------")
                    printer.print_to_file("")
                    printer.print_to_file("")

                    # Find the best configuration
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_config = [model, optimizer, size, rate, decay, prob, "%.4f" % accuracy, "%.4f" % f1_score]

    # Convert global result list into Pandas Matrix for better visualization
    config.train_results.append(best_config)

    return best_config


def fit_and_score(x, y, embeddings, config):

    # Build Model for Cross Validation with configuration of current CV setting
    tagger = SRLModel(config, embeddings, config.label_dict.size())

    # Measurements
    accuracy = 0
    f1_score = 0

    # Define the number of folds
    # Change the number of folds to k = 9, since otherwise array split does not result in an equal division
    k_folds = 10

    # Split data into k folds
    x_folds = np.split(x, indices_or_sections=k_folds, axis=0)
    y_folds = np.split(y, indices_or_sections=k_folds, axis=0)

    # Do k-Fold Cross Validation
    for k in range(k_folds):

        # Training Data
        x_train = []
        y_train = []

        # Development Data
        x_dev = []
        y_dev = []

        for n in range(k_folds):

            if k != n:
                for x_fold, y_fold in zip(x_folds[n], y_folds[n]):
                    x_train.append(x_fold)
                    y_train.append(y_fold)
            else:
                for x_fold, y_fold in zip(x_folds[n], y_folds[n]):
                    x_dev.append(x_fold)
                    y_dev.append(y_fold)

        # Convert list into Numpy Array
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_dev = np.array(x_dev)
        y_dev = np.array(y_dev)


        # Check whether the folder "/model.weights" for the model output exists
        # and create a new folder if needed
        if not os.path.exists(config.model_output):
            os.makedirs(config.model_output)

        # Create a new instance of the Model
        model = copy.deepcopy(tagger)
        model.build()

        # Train Model with different datasets
        acc, f1, loss = model.train(x_train, x_dev, y_train, y_dev)
        # acc, pre, rec, f1 = model.evaluate(x_test, y_test)

        # nested xfold
        # Use the best model out of the 9 folds to evalutate against the

        # Summarize measurements to average them in the end
        accuracy += acc
        f1_score += f1

        # Reset Model Graph
        tf.reset_default_graph()

        # Delete model copy and output folder with previously saved models
        shutil.rmtree(config.model_output)
        del model


    # Delete Model
    del tagger

    # Return average of Accuracy and F1 Score based on Development Set
    return (accuracy / k_folds), (f1_score / k_folds)


def test_and_score(x, y, x_test, y_test, cv_results, embeddings, config):

    test_results = []

    # Reset Model Graph
    tf.reset_default_graph()

    # Split Dataset into Training and Development Set
    x_train, x_dev, y_train, y_dev = dev_split(x, y, test_size=0.2, random_state=42)


    for result in cv_results:

        config.model_type = result[0]
        config.learning_method = result[1]
        config.hidden_size = result[2]
        config.learning_rate = result[3]
        config.lr_decay = result[4]
        config.dropout = 1.0

        dev_acc = result[6]
        dev_f1 = result[7]

        # Check whether the folder "/model.weights" for the model output exists
        # and create a new folder if needed
        if not os.path.exists(config.model_output):
            os.makedirs(config.model_output)

        # Build Model for Cross Validation with configuration of current CV setting
        tagger = SRLModel(config, embeddings, config.label_dict.size())
        tagger.build()

        evaluation = ["word", "entity"]

        # Train Model with different datasets
        _, _, loss = tagger.train(x_train, x_dev, y_train, y_dev)

        for eval in evaluation:

            config.eval_level = eval

            # Evaluate Model on Test Set
            acc, pre, rec, f1 = tagger.evaluate(x_test, y_test, test_mode=True)

            test_results.append(
                [config.model_type, config.learning_method, config.hidden_size, config.learning_rate, config.lr_decay, config.dropout, result[5], dev_acc, dev_f1,
                 "%.4f" % acc, "%.4f" % f1])

        # Reset Model Graph
        tf.reset_default_graph()

        # Delete model copy and output folder with previously saved models
        shutil.rmtree(config.model_output)
        del tagger


    # Print the final results with model parameters and accuracy, f1-score
    # Convert global result list into Pandas Matrix for better visualization
    df = pd.DataFrame(test_results,
                      columns=["Model", "Optimizer", "Hidden Size", "Learning Rate", "LR Decay", "Train Dropout", "Test Dropout",
                               "Dev Accuracy", "Dev F1-Score", "Test Accuracy", "Test F1-Score"])

    printer.print_to_file("")
    printer.print_to_file(df)
    printer.print_to_file("")
