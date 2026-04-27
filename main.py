import tokenizer
import generator
import config
import torch
import torch.optim as optim

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = config.Config.load('config.json')

# Read data
with open('data.txt', encoding='utf-8') as f:
    data = f.read()

# Load tokenizer
tkz = tokenizer.Tokenizer()
tkz.load(data)
tkz.set_batch_data(data)

# Create model
model = generator.Generator(cfg.hidden_state, cfg.features, tkz.vocab_count, tkz.pad).to(device)

# Run model
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

prev_epoch_loss = 0.0
for i in range(cfg.epochs):
    batch_iter = tkz.create_batches(cfg.batch_size)

    mean_epoch_loss = model.train(batch_iter, optimizer)
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




