import re
import random
from itertools import batched
from typing import Generator

class Tokenizer:

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace('İ', 'i').replace('I', 'ı').lower()
        return text

    @staticmethod
    def extract(text: str) -> list[str]:
        text = re.findall(r'[a-zçğıöşü]+', text)
        return text

    def __init__(self):
        self.unk = 0
        self.eow = 1
        self.sow = 2
        self.pad = 3

        self.banned_tokens = [self.unk, self.pad, self.sow]

        self.letters = {'<unk>': 0, '</w>': 1, '<w>': 2, '<pad>': 3}
        self.letters_t = {}
        self.vocab_count = 4

        self.encoded_train_data = None
        self.encoded_val_data = None
        self.encoded_test_data = None

    def load(self, text: str) -> None:
        text = Tokenizer.clean_text(text)
        text = Tokenizer.extract(text)

        for word in text:
            length = len(word)
            for i in range(length):
                char = word[i]
                if char not in self.letters:
                    self.letters[char] = self.vocab_count
                    self.vocab_count += 1

        self.letters_t = {idx: word for word, idx in self.letters.items()}

    def encode(self, word: str, eow: bool = True) -> list[int]:
        encoded_pieces = [self.sow] + [self.letters.get(ch, self.unk) for ch in word] + ([self.eow] if eow else [])
        return encoded_pieces

    def decode(self, encodings: list[int]) -> str:
        decoded_ = []
        for encoding in encodings:
            token = self.letters_t[encoding]
            if token == '</w>':
                break
            elif token == '<w>':
                continue
            decoded_.append(token)

        return ''.join(decoded_)

    def padding(self, text: list[int], count: int) -> list[int]:
        return text + [self.pad] * count

    def set_batch_data(self, train: str, val: str, test: str) -> None:
        train = Tokenizer.clean_text(train)
        train = Tokenizer.extract(train)
        self.encoded_train_data = [self.encode(token, eow=True) for token in train]

        val = Tokenizer.clean_text(val)
        val = Tokenizer.extract(val)
        self.encoded_val_data = [self.encode(token, eow=True) for token in val]

        test = Tokenizer.clean_text(test)
        test = Tokenizer.extract(test)
        self.encoded_test_data = [self.encode(token, eow=True) for token in test]
    def create_batches(self, batch_type: str = 'train', batch_size: int = 32, shuffle: bool = True) \
            -> Generator[tuple[list[list[int]], list[list[int]]], None, None]:

        batch_types = {
            'train': self.encoded_train_data,
            'val': self.encoded_val_data,
            'test': self.encoded_test_data
        }

        if batch_type not in batch_types: raise Exception('Invalid batch type.')

        selected_data = list(batch_types[batch_type])
        selected_data.sort(key=len)

        raw_batches = batched(selected_data, batch_size)
        raw_batches = list(raw_batches)
        if batch_type == 'train' and shuffle: random.shuffle(raw_batches)

        for raw_batch in raw_batches:
            seq_len = len(max(raw_batch, key=len))
            batch_x = []
            batch_y = []
            for word in raw_batch:
                tokens = self.padding(word, seq_len - len(word))
                batch_x.append(tokens[:-1])
                batch_y.append(tokens[1:])

            yield batch_x, batch_y

    def tokenize_pred(self, text: str) -> list[int]:
        text = Tokenizer.clean_text(text)
        return self.encode(text, eow=False)