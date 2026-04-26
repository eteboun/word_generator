import json

class Config:

    parameters = {"features": int, "hidden_state": int,
                  "batch_size": int, "epochs": int, "lr": (int, float), "step_size": (int, float), "gamma": (int, float),
                  "temperature": (int, float), "p": (int, float), "freq_penalty": (int, float), "n": int}

    def __init__(self, features, hidden_state,
                 batch_size, epochs, lr, step_size, gamma,
                 temperature, p, freq_penalty, n):

        self.features = features
        self.hidden_state = hidden_state
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.temperature = temperature
        self.p = p
        self.freq_penalty = freq_penalty
        self.n = n

    @classmethod
    def check_params(cls, params):

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
    def load(cls, address):
        with open(address) as cfg:
            params = json.load(cfg)

        cls.check_params(params)
        return cls(**params)