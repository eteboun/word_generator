import json
from typing import Self

class Config:
    parameters = {}

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

    @classmethod
    def load(cls, address: str) -> Self:
        with open(address) as cfg:
            params = json.load(cfg)

        cls.check_params(params)
        return cls(**params)

    def get_elements(self):
        return self.__dict__.copy()

class ModelConfig(Config):
    parameters = {"features": int, "hidden_state": int}

    def __init__(self, features: int, hidden_state: int) -> None:
        self.features = features
        self.hidden_state = hidden_state

class TrainConfig(Config):
    parameters = {"batch_size": int, "epochs": int,
                  "lr": (int, float)}

    def __init__(self, batch_size: int, epochs: int, lr: int | float) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

class PredictConfig(Config):
    parameters = {"temperature": (int, float), "p": (int, float),
                  "freq_penalty": (int, float), "n": int}

    def __init__(self, temperature: int | float, p: int | float,
                 freq_penalty: int | float, n: int) -> None:
        self.temperature = temperature
        self.p = p
        self.freq_penalty = freq_penalty
        self.n = n

class PathConfig(Config):
    parameters = {"new_model": str, "data_name": str,
                  "tokenizer": str, "curr_model": str}

    def __init__(self, new_model: str, data_name: str,
                 tokenizer: str, curr_model: str) -> None:
        self.new_model = new_model
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.curr_model = curr_model

class InitConfig(Config):
    parameters = {"model": dict, "train": dict}
    def __init__(self, model: ModelConfig, train: TrainConfig) -> None:
        self.model = model
        self.train = train

    @classmethod
    def load(cls, address: str) -> Self:
        with open(address) as cfg:
            params = json.load(cfg)

        cls.check_params(params)

        ModelConfig.check_params(params["model"])
        TrainConfig.check_params(params["train"])

        model = ModelConfig(**params["model"])
        train = TrainConfig(**params["train"])
        return cls(model=model, train=train)

class ForwardTrainConfig(Config):
    parameters = {"train": dict, "paths": dict}
    def __init__(self, train: TrainConfig, paths: PathConfig) -> None:
        self.train = train
        self.paths = paths

    @classmethod
    def load(cls, address: str) -> Self:
        with open(address) as cfg:
            params = json.load(cfg)

        cls.check_params(params)

        TrainConfig.check_params(params["train"])
        PathConfig.check_params(params["paths"])

        train = TrainConfig(**params["train"])
        paths = PathConfig(**params["paths"])
        return cls(train=train, paths=paths)