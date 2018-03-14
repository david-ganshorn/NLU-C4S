# ======================================================================================================================
# Copyright © 2017 David Ganshorn. All rights reserved.
# Created by David Ganshorn on 08/04/2017.
#
# Last Updated by David Ganshorn on 08/04/2017.
#
# Comment:
# Had to implement own function of dictionary, since tensorflow.contrib.learn
# cuts tokens like '<s>', '</s>' and <UNK> into 's' and 'UNK'
# reuse of dictionary implementation by Luheng He
#
# ======================================================================================================================

from utils import constants

""""
Bidirectional dictionary that maps between words and ids.
"""

class Dictionary(object):
    def __init__(self, unknown_token=None):
        self.str2idx = {}
        self.idx2str = []

        self.accept_new = True

        self.unknown_token = constants.unknown_token                # Token that indicates that a word or label is unknown
        self.unknown_label_token = constants.unknown_label          # Token that indicates that a label is unknown
        self.start_token = constants.start_token                    # Token that indicates the beginning of a sequence
        self.end_token = constants.end_token                        # Token that indicates the end of a sequence

        self.unk_token_id = None
        self.unknown_label_token_id = None
        self.start_token_id = None
        self.end_token_id = None

        # if unknown_token is None:
        #     self.initialize_unknown_token()

    def initialize_unknown_token(self):
        self.unk_token_id = self.add(self.unknown_token)

    def initialize_tokens(self):
        self.initialize_unknown_token()
        self.unknown_label_token_id = self.add(self.unknown_label_token)
        self.start_token_id = self.add(self.start_token)
        self.end_token_id = self.add(self.end_token)

    def add(self, new_str):
        if not new_str in self.str2idx:
            if self.accept_new:
                self.str2idx[new_str] = len(self.idx2str)
                self.idx2str.append(new_str)
            else:
                if self.unk_token_id is None:
                    raise LookupError(
                        'Trying to add new token to a freezed dictionary with no pre-defined unknown token: ' + new_str)
                return self.unk_token_id

        return self.str2idx[new_str]

    def add_all(self, str_list):
        return [self.add(s) for s in str_list]

    def get_index(self, input_str):
        if input_str in self.str2idx:
            return self.str2idx[input_str]
        return None

    def size(self):
        return len(self.idx2str)

    def save(self, filename):
        with open(filename, 'w') as f:
            for s in self.idx2str:
                f.write(s + '\n')
            f.close()

    def load(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    self.add(line)
            f.close()


