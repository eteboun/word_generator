import json
from typing import Self

class TrainConfig:

    parameters = {"features": int, "hidden_state": int,
                  "batch_size": int, "epochs": int, "lr": (int, float)}

    def __init__(self, features, hidden_state,
                 batch_size, epochs, lr):

        self.features = features
        self.hidden_state = hidden_state
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

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
        return {
            "features": self.features,
            "hidden_state": self.hidden_state,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr
        }