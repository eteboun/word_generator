import torch
import tokenizer.tokenizer as tokenizer
import config.config as config
from model.generator import Model

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = config.PredictConfig.load(f'./config/predict_cfg.json')

# Load tokenizer
tkz = tokenizer.Tokenizer().load(f"./tokenizer/{cfg.tokenizer}.json")

# Load model
model_general_data = torch.load(f"./model_saves/{cfg.model_name}/info.pt", map_location=device)

model_cfg = config.ModelConfig(**model_general_data["model_config"])
model = Model(model_cfg, tkz.vocab_count, tkz.pad).to(device)

model_state_dict = model_general_data["model"]
model.load_state_dict(model_state_dict)

# Predict
while True:
    prompt = input("> ")
    if prompt == "exit": break

    ids = tkz.get_prompt(prompt)
    predictions = model.generate(encoded_input=ids,
                                 eow_id=tkz.eow,
                                 temperature=cfg.temperature,
                                 p=cfg.p,
                                 freq_penalty=cfg.freq_penalty,
                                 n=cfg.n)

    word = tkz.decode(ids + predictions)
    print(word)