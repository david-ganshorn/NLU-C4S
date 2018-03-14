
import datetime
import os.path as path

# holds constants that are used different files

time = str(datetime.datetime.now())                                     # Time of Start

root_folder = path.abspath(path.join(__file__, "../.."))                # Path to the root folder of the project
data_folder = root_folder + "/data"                                     # Path to the folder where the data files are stored
summary_folder = data_folder + "/summary"

training_data = "1050_dataset_sql_BIO_tags.txt"                         # Filename of the training data
embedding_data_small = "glove.42B.300d.txt"                             # Name of the file where the word embeddings are stored
embedding_data_big = "glove.840B.300d.txt"                              # Name of the file where the word embeddings are stored
char_translation_data = "dataset_translate_char_level.txt"              # Filename for training data with question pairs (English Question/ SQL Query)
# 00-dataset_translate_simple_word_level.txt
# 20-dataset_translate_basic_word_level.txt
# 30-dataset_translate_complex_word_level.txt
word_translation_data = "00-dataset_translate_simple_word_level.txt"        # Filename for training data on a word level
full_word_translation_data = "30-dataset_translate_complex_word_level.txt"  # Filename for training data on a word level

unknown_token = "<UNK>"                                                 # Token that indicates that a word or label is unknown
unknown_label = "o"                                                     # Token that indicates that a label is unknown
start_token = "<s>"                                                     # Token that indicates the beginning of a sequence
end_token = "</s>"                                                      # Token that indicates the end of a sequence


tag_test_set = [954, 37, 585, 153, 667, 696, 962, 431, 581, 807, 414, 408, 744, 1019, 515, 60, 562, 543, 805, 989,
                889, 911, 526, 1032, 752, 892, 244, 1039, 24, 446, 189, 13, 90, 687, 1025, 734, 325, 518, 354, 52,
                709, 762, 66, 836, 206, 866, 875, 210, 551, 846, 759, 299, 688, 334, 537, 587, 887, 212, 501, 466,
                70, 789, 800, 870, 538, 955, 640, 746, 1013, 941, 319, 531, 533, 103, 614, 516, 264, 487, 797, 100,
                638, 548, 1018, 694, 146, 315, 782, 1040, 248, 739, 648, 480, 207, 495, 50, 16, 765, 547, 522, 535,
                134, 282, 317, 831, 192, 314, 73, 364, 144, 175, 917, 1010, 749, 274, 855, 201, 665, 900, 1022, 151,
                751, 914, 567, 361, 356, 182, 860, 619, 42, 40, 275, 202, 676, 170, 496, 257, 295, 906, 563, 327,
                445, 190, 730, 880, 668, 549, 76, 620, 360, 600]
