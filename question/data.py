
import numpy as np

from utils import constants, dictionary as dict


# Prepare & Build the dataset with training values
def build_words(questions, configuration):

    config = configuration

    # Build an empty dictionary for words
    word_dict = dict.Dictionary()

    print("Dict")
    print(word_dict.idx2str)

    # Placeholder for returned dataset
    x = []

    for question in questions:

        # Temporary row
        row = []

        for token in question.tokens:
            # First add "new" word to dict
            token_id = word_dict.add(token)

            # Add token_id to the data
            row.append(token_id)

        # If padding is enabled - fill sequence with unknown tokens
        if config.use_word_padding:
            while len(row) < config.max_sequence_length:
                row.append(word_dict.get_index(constants.unknown_token))

        # Add new row to the data
        x.append(row)

    # Transform x to numpy array
    x = np.array(x)

    # Set configuration value for word_dict
    config.word_dict = word_dict

    return x


# Prepare & Build the dataset with target values
def build_labels(questions, configuration):

    config = configuration

    # Build an empty dictionary for words
    label_dict = dict.Dictionary()

    print("label_dict")
    print(label_dict.idx2str)

    # Add unknown label marker if label padding is enabled
    if config.use_label_padding:
        label_dict.add(constants.unknown_label)

    # Placeholder for returned labels
    y = []

    for question in questions:

        # Temporary row
        row = []

        for label in question.semantic_labels:
            # First add "new" word to dict
            label_id = label_dict.add(label)

            # Add token_id to the data
            row.append(label_id)

        # Add new row to the data
        y.append(row)

    # Transform y to numpy array
    y = np.array(y)

    # Set configuration value for word_dict
    config.label_dict = label_dict

    return y


def build_types(questions, type, configuration):

    config = configuration

    y = []

    # Build an empty dictionary for words
    label_dict = dict.Dictionary()

    print("label_dict")
    print(label_dict.idx2str)

    for question in questions:

        if type == "cnn_question":
            # Fill dictionary with values
            label_dict.add(question.question_type)
        else:
            # Fill dictionary with values
            label_dict.add(question.answer_type)

    config.label_dict = label_dict


    # Build label data
    for question in questions:

        row = []
        for label in config.label_dict.idx2str:
            if type == "cnn_question":
                if label == question.question_type:
                    row.append(1)
                else:
                    row.append(0)
            else:
                if label == question.answer_type:
                    row.append(1)
                else:
                    row.append(0)

        y.append(row)

    y = np.array(y)

    return y
