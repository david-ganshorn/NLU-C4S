
import datetime
from utils import constants, dictionary as dict


class Variables:

    # holds variables that are used and changed in different files
    def __init__(self):

        self.summary_file_name = "tagger_" + str(datetime.datetime.now()) + ".txt"  # Filename of the output summary file

        self.training_data = constants.training_data                # Name of the file where the training data is stored
        self.test_questions = "sql_BIO_questions.txt"
        self.embedding_data = constants.embedding_data_small        # Name of the file where the word embeddings are stored

        self.result_folder = constants.root_folder + "/model/"      # Path to the folder where the tagger files are stored
        self.summary_folder = self.result_folder + "/summary"       # Path to the folder where Printer Output is stored
        self.model_data = self.result_folder  # + "/saved_model"       # Path to the folder where the models will be stored
        self.model_output = self.model_data  #+ "/model.weights/"     # Path to the folder where the models will be stored
        # self.model_output = "/Users/davidganshorn/Desktop/model.weights/"

        self.train_results = []                                     # Array to store all results during training
        self.best_train_config = []                                 # Store the configuration with the best configuration
        self.accuracy = 0
        self.f1_score = 0

        self.word_dict = dict.Dictionary                            # Dictionary containing (word: key) pairs
        self.label_dict = dict.Dictionary                           # Dictionary containing (label: key) pairs
        self.semantic_labels = []                                   # Array with distinct labels

        self.table_name_dict = dict.Dictionary                      # Array with distinct table names
        self.question_types_dict = dict.Dictionary                  # Array with distinct question types
        self.answer_types_dict = dict.Dictionary                    # Array with distinct answer types

        self.use_sequence_marker = False                            # Use Start <s> and End </s> marker for sequences
        self.use_word_padding = False                               # Use padding for word sequences (default: True)
        self.use_label_padding = False                              # Use padding for label sequences
        self.group_by_seq_length = False                            # Group training instances by their length of words

        self.max_sequence_length = 0                                # Length of longest question

        self.chars = False                                          # if char embedding, training is 3.5x slower on CPU

        self.max_iter = None                                        # if not None, max number of examples

        self.batch_size = 25                                        # Defines the size of batches that are being fed to the model during training (default: 80)
        self.dev_batch_size = 10                                    # Batch Size for Development Set

        self.hidden_size = 100                                      # Number of Hidden Units
        self.max_grad_norm = 1.0
        self.model_type = "bi-lstm"                                 # "ellman", "gru", "rnn", "lstm", or "bi-lstm"

        self.train_embeddings = False                               #
        self.crf = True                                             # Conditional Random Fields

        self.learning_method = "adam"                               # Optimization Method
        self.lr = 0.01                                              # Learning Rate
        self.lr_decay = 0.95                                         #
        self.clip = -1                                              # if negative, then no clipping
        self.reload = False                                         #
        self.dropout = 0.50                                          # Regularization technique for reducing overfitting

        # This parameter is required to evaluate predictions based on a word or entity level
        # 'Word':   We look at each word independently
        # 'Entity': We take compound words into consideration
        self.eval_level = "entity"

        # This parameter is required for multi label targets ('macro', 'micro')
        # 'Macro':  Calculate metrics for each label, and find their unweighted mean
        # 'Micro':  Calculate metrics globally by counting the total true positives, false negatives and false positives
        self.score_method = "macro"                                 #

        # This parameter defines the scoring metrics
        # 'Relaxed':    Evaluate only the tokens
        # 'Strict':     Evaluate the tokens and their cohesiveness/order, e.g., "North America" =>  "B_where mea" is wrong
        # self.eval_method = "relaxed"                              #
        self.eval_method = "strict"

        self.n_epochs = 50                                          # Number of epochs to
        self.n_epoch_no_imprv = 1                                   # Amount of epochs w/o improvement to abort training
        self.n_epoch_stagnating = 5                                 # Number of epochs where result is constant

        self.embedding_size = 300                                   # Dimensionality of character embedding (default (GloVe): 300)


    def print_configuration(self):
        print()






