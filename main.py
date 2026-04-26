import tokenizer
import generator
import config
import torch
import torch.optim as optim
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        loss = func(*args, **kwargs)
        end = time.time()

        print(f'{func.__name__} train step took {(end - start) * 1000:.3f} ms')
        return loss
    return wrapper

def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'epoch started.')
        loss = func(*args, **kwargs)
        print(f'epoch ended.')
        return loss
    return wrapper

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = config.Config.load('config.json')

# Read data
with open('data.txt', encoding='utf-8') as f:
    data = f.read()

# Load tokenizer
tkz = tokenizer.Tokenizer()
tkz.load(data)

# Create model
model = generator.Generator(cfg.hidden_state, cfg.features, tkz.vocab_count, tkz.pad).to(device)

@logger
@timer
def train(model):
    epoch_loss = 0.0
    count = 0

    for x, y in tkz.batch(data, cfg.batch_size):
        optimizer.zero_grad()

        inputs = torch.tensor(x, dtype=torch.long, device=device)
        targets = torch.tensor(y, dtype=torch.long, device=device)

        loss = model(inputs, targets)

        loss.backward()
        optimizer.step()

        count += 1
        epoch_loss += loss.item()

    calc_loss = epoch_loss / count
    return calc_loss


# Run model
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

prev_epoch_loss = 0.0
for i in range(cfg.epochs):

    mean_epoch_loss = train(model)
    scheduler.step()

    print(f'epoch {i+1} loss: {mean_epoch_loss:.4f}')

    if i != 0:
        loss_diff = prev_epoch_loss - mean_epoch_loss
        print(f'loss change: ↓{loss_diff:.4f}')

    prev_epoch_loss = mean_epoch_loss

# Predictions
while 1:

    inputs = tkz.tokenize_pred(input('Enter a word: '))
    tensor = torch.tensor(inputs, dtype=torch.long, device=device).unsqueeze(0)
    encoded_ = model.pred(tensor, tkz.eow, tkz.banned_tokens, n=cfg.n, temperature=cfg.temperature, p=cfg.p, freq_penalty=cfg.freq_penalty)[0][1:]
    word = tkz.decode(encoded_)
    print(word)




