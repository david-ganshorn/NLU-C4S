
import os
import tensorflow as tf

from utils import dictionary as dict


# ======================================================================================================================
# Class Question
# ======================================================================================================================
class Question:
    def __init__(self, question_id, question, tokens, pos, ner, table, question_type, answer_type, semantic_labels, translated_sql):
        self.id = question_id
        self.question = question
        self.tokens = tokens
        self.pos = pos
        self.ner = ner
        self.table_name = table
        self.question_type = question_type
        self.answer_type = answer_type
        self.semantic_labels = semantic_labels

        self.translated_sql = translated_sql


# ======================================================================================================================
# Class QuestionLoader
# ======================================================================================================================
class QuestionLoader:
    def __init__(self, path, file):
        self.data_folder = path
        self.questions = []
        self.file_name = file

    # Load Data from Disk
    def load_data(self):
        f = tf.gfile.GFile(os.path.join(self.data_folder, self.file_name), "r")

        counter = 0
        for line in f:
            if counter > 0:
                line = line.strip()

                # Replace all tabs with a whitespace
                line = str(line)
                processed_line = line.replace("\t", " ")

                # Divide questions and labels which are separated by '|'
                question = processed_line.split(" | ")

                self.questions.append(question)

            counter += 1

        f.close()

        return self.questions


    # Load Test Data from Disk
    # Todo
    # define test data and load from disk
    def load_test_data(self):
        f = tf.gfile.GFile(os.path.join(self.data_folder, self.file_name), "r")

        counter = 0
        for line in f:
            if counter > 0:
                line = line.strip()

                # Replace all tabs with a whitespace
                line = str(line)
                processed_line = line.replace("\t", " ")

                # Divide questions and labels which are separated by '|'
                question = processed_line.split(" | ")

                self.questions.append(question)

            counter += 1

        f.close()

        return self.questions


# ======================================================================================================================
# Class QuestionGenerator
# ======================================================================================================================
class QuestionGenerator:
    def __init__(self, data, configuration):
        self.questions = data
        self.config = configuration

    # Pre Process the loaded input
    def pre_processing(self):

        processed_questions = []

        # Build an empty dictionary for question types
        question_type_dict = dict.Dictionary()

        # Build an empty dictionary for answer types
        answer_type_dict = dict.Dictionary()

        # Build an empty dictionary for table names
        table_dict = dict.Dictionary()

        for item in self.questions:

            # Extract the id of the question in the dataset
            question_id = item[0]

            # Extract the questions from loaded inputs and convert to lower case
            question = item[1]
            question = question.lower()

            # Extract the table name lower case
            # and add table name to dictionary
            table_name = item[2]
            table_name = table_name.lower()
            table_dict.add(table_name)

            # Extract the question type from loaded inputs and convert to lower case
            # and add question type to dictionary
            question_type = item[3]
            question_type = question_type.lower()
            question_type_dict.add(question_type)

            # Extract the answer type from loaded inputs and convert to lower case
            # and add answer type to dictionary
            answer_type = item[4]
            answer_type = answer_type.lower()
            answer_type_dict.add(answer_type)

            # Extract the target labels from loaded inputs and convert to lower case
            semantic_labels = item[5]
            semantic_labels = semantic_labels.lower()

            # Tokenize Questions and Labels
            question_tokens, semantic_label_tokens = self.token_generator(question, semantic_labels)

            # Receive POS Tag List for tokens
            pos = self.pos_generator(question_tokens)

            # Get list of unique labels
            # We do not use a dictionary because "<UNK>", "o", "<s>", "</s>" would automatically be added when initializing the dict
            for token in semantic_label_tokens:
                if not self.config.semantic_labels.__contains__(token):
                    self.config.semantic_labels.append(token)

            if len(question_tokens) != len(semantic_label_tokens):
                print("Error: Question", str(question_id), "has different lengths!")

            # Create Question Object
            q = Question(question_id, question, question_tokens, pos, "ner", table_name, question_type, answer_type, semantic_label_tokens, None)
            processed_questions.append(q)

            # print(q.id, q.tokens, q.semantic_labels)

        # Call the method to find the longest sequence
        self.max_sequence_length(processed_questions)

        self.config.table_name_dict = table_dict
        self.config.question_types_dict = question_type_dict
        self.config.answer_types_dict = answer_type_dict

        return processed_questions


    # Pre Process the loaded input
    def translation_pre_processing(self):

        processed_questions = []

        difference = 0

        for item in self.questions:

            # Extract the id of the question in the dataset
            question_id = item[0]

            # Extract the questions from loaded inputs and convert to lower case
            question = item[1]
            question = question.lower()

            # Extract the question type from loaded inputs and convert to lower case
            # and tokenize them on a character level
            # and add question type to dictionary
            translated_sql = item[2]
            translated_sql = translated_sql.lower()

            if self.config.translation_level == "char":
                # Tokenize Questions and Translations on a character level
                question_tokens = list(question)
                translated_sql_tokens = list(translated_sql)
            else:
                # Tokenize Questions and Translations on a word level
                question_tokens, translated_sql_tokens = self.token_generator(question, translated_sql)

            # Create Question Object
            q = Question(question_id, question, question_tokens, None, None, None, None, None, None, translated_sql_tokens)
            processed_questions.append(q)

            difference += len(translated_sql_tokens) - len(question_tokens)

        # print("The average difference is:", str(difference / 515))

        # Call the method to find the longest sequence
        question_sequence_length = 0
        sql_sequence_length = 0

        for question in processed_questions:
            if len(question.tokens) > question_sequence_length:
                question_sequence_length = len(question.tokens)

            if len(question.translated_sql) > sql_sequence_length:
                sql_sequence_length = len(question.translated_sql)

        self.config.max_sequence_length = sql_sequence_length

        return processed_questions


    # Pre Process the loaded test input
    def process_test_data(self):

        processed_questions = []

        for item in self.questions:

            # Extract the questions from loaded inputs and convert to lower case
            question = item[0]
            question = question.lower()

            processed_questions.append(question)

        return processed_questions


    # Determine the length of the longest sequence
    def max_sequence_length(self, processed_questions):
        sequence_length = 0

        for question in processed_questions:
            if len(question.tokens) > sequence_length:
                sequence_length = len(question.tokens)

        self.config.max_sequence_length = sequence_length


    # Generate Tokens from Text
    def token_generator(self, sentence_input, label_input):

        # Alternative 1 - Based on NLTK
        # Bad, since, for example,  also 'T&E' will be splitted
        # return nltk.word_tokenize(sentence_input)

        # Alternative 2 - Based on Python
        word_tokens = sentence_input.split(" ")
        label_tokens = label_input.split(" ")

        return word_tokens, label_tokens

    # Generate POS Tagging based on Tokens
    def pos_generator(self, pos_input):

        pos = []

        # POS Tagging
        # pos_tagged = nltk.pos_tag(pos_input)

        # for tag in pos_tagged:
        #     pos.append(tag[1])

        return pos

    # Pre Process the loaded input
    def text_pre_processing(self):

        processed_questions = []

        # Build an empty dictionary for question types
        question_type_dict = dict.Dictionary()

        # Build an empty dictionary for answer types
        answer_type_dict = dict.Dictionary()

        for item in self.questions:

            # Extract the id of the question in the dataset
            question_id = item[0]

            # Extract the questions from loaded inputs and convert to lower case
            question = item[1]
            question = question.lower()

            # Extract the question type from loaded inputs and convert to lower case
            # and add question type to dictionary
            question_type = item[2]
            question_type = question_type.lower()
            question_type_dict.add(question_type)

            # Extract the answer type from loaded inputs and convert to lower case
            # and add answer type to dictionary
            answer_type = item[3]
            answer_type = answer_type.lower()
            answer_type_dict.add(answer_type)

            # Extract the target labels from loaded inputs and convert to lower case
            semantic_labels = item[4]
            semantic_labels = semantic_labels.lower()

            # Tokenize Questions and Labels
            question_tokens, semantic_label_tokens = self.token_generator(question, semantic_labels)

            # Receive POS Tag List for tokens
            pos = self.pos_generator(question_tokens)

            # Get list of unique labels
            for token in semantic_label_tokens:
                if not self.config.semantic_labels.__contains__(token):
                    self.config.semantic_labels.append(token)

            # Create Question Object
            q = Question(question_id, question, question_tokens, pos, "ner", question_type, answer_type, semantic_label_tokens)
            processed_questions.append(q)

            # print(q.id)
            # print(q.question)
            # print(q.question_type)
            # print(q.answer_type)
            # print()

        # Call the method to find the longest sequence
        self.max_sequence_length(processed_questions)

        self.config.question_types_dict = question_type_dict
        self.config.answer_types_dict = answer_type_dict

        return processed_questions
