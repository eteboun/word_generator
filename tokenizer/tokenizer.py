import re
import random
import json
from itertools import batched
from typing import Generator, Self

class Tokenizer:

    parameters = {'vocab_count': int, 'letters': dict,
                  'letters_t': dict}

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace('İ', 'i').replace('I', 'ı').lower()
        return text

    @staticmethod
    def extract(text: str) -> list[str]:
        text = re.findall(r'[a-zçğıöşü]+', text)
        return text

    @classmethod
    def load(cls, address: str) -> Self:
        with open(address, 'r') as f:
            state_dict = json.load(f)

        cls.check_params(state_dict)
        letters_t = {int(k): v for k, v in state_dict["letters_t"].items()}

        state_dict["letters_t"] = letters_t
        return cls(**state_dict)

    @classmethod
    def check_params(cls, params: dict) -> None:
        for k, v in params.items():
            required_params_instance = cls.parameters.get(k, None)
            if required_params_instance is None:
                raise ValueError(f"{k} is not a valid parameter.")
            if not isinstance(v, required_params_instance):
                raise TypeError(f"{v} is an unexpected value for the parameter {k} ({required_params_instance} expected).")

        for k in cls.parameters:
            if k not in params:
                raise ValueError(f"{k} is missing.")

    def __init__(self, letters = None, letters_t = None, vocab_count = None):

        self.unk = 0
        self.eow = 1
        self.sow = 2
        self.pad = 3

        self.banned_tokens = [self.unk, self.pad, self.sow]

        if letters is None: self.letters = {'<unk>': 0, '</w>': 1, '<w>': 2, '<pad>': 3}
        else: self.letters = letters
        if letters_t is None: self.letters_t = {}
        else: self.letters_t = letters_t
        if vocab_count is None: self.vocab_count = 4
        else: self.vocab_count = vocab_count

        self.encoded_train_data = None
        self.encoded_val_data = None
        self.encoded_test_data = None

    def create_vocab(self, text: str) -> None:
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

    def get_prompt(self, text):
        if not text:
            return self.encode('', eow=False)
        text = Tokenizer.clean_text(text)
        text = Tokenizer.extract(text)
        return self.encode(text[0], eow=False)

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

    def save(self, address: str) -> None:

        state_dict = {
            'vocab_count': self.vocab_count,
            'letters': self.letters,
            'letters_t': self.letters_t,
        }

        with open(address, 'w') as f:
            json.dump(state_dict, f)


