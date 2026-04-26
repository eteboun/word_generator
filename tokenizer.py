import re
import random
from itertools import batched

class Tokenizer:
    def __init__(self):
        self.unk = 0
        self.eow = 1
        self.sow = 2
        self.pad = 3

        self.banned_tokens = [self.unk, self.pad, self.sow]

        self.letters = {'<unk>': 0, '</w>': 1, '<w>': 2, '<pad>': 3}
        self.letters_t = {}
        self.vocab_count = 4

    def load(self, text):
        text = text.replace('İ', 'i').replace('I', 'ı').lower()
        text = re.findall(r'[^\W_]+', text)

        for word in text:
            length = len(word)
            for i in range(length):
                char = word[i]
                if char not in self.letters:
                    self.letters[char] = self.vocab_count
                    self.vocab_count += 1

        self.letters_t = {idx: word for word, idx in self.letters.items()}

    def tokenize_batch(self, text, seq_len):

        seq_len -= 2

        tokenized_ = [self.sow]
        for char in text:

            if seq_len <= 0:
                raise Exception('Sequence too long.')

            tokenized_.append(self.letters.get(char, self.unk))
            seq_len -= 1
        tokenized_.append(self.eow)

        for _ in range(seq_len):
            tokenized_.append(self.pad)

        return tokenized_

    def tokenize_pred(self, text):
        text = text.replace('İ', 'i').replace('I', 'ı').lower()

        tokenized_ = [self.sow]
        for char in text:
            tokenized_.append(self.letters.get(char, self.unk))

        return tokenized_

    def batch(self, text, batch_size):

        text = text.replace('İ', 'i').replace('I', 'ı').lower()
        text = re.findall(r'[^\W_]+', text)
        random.shuffle(text)

        raw_batches = batched(text, batch_size)
        for raw_batch in raw_batches:
            seq_len = len(max(raw_batch, key=len)) + 2
            batch_x = []
            batch_y = []
            for word in raw_batch:
                tokens = self.tokenize_batch(word, seq_len)
                batch_x.append(tokens[:-1])
                batch_y.append(tokens[1:])

            yield batch_x, batch_y

    def decode(self, encodings):
        decoded_ = []
        for encoding in encodings:
            token = self.letters_t[encoding]
            if token.endswith('</w>'):
                decoded_.append(token[:-4])
                break
            decoded_.append(token)

        return ''.join(decoded_)

