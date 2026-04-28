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
tkz.set_batch_data(data=data, train=0.7, val=0.2, test=0.1)

# Create model
model = generator.Generator(cfg.hidden_state, cfg.features, tkz.vocab_count, tkz.pad).to(device)

# Run model
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

prev_train_loss = 0.0
for i in range(cfg.epochs):
    batch_iter = tkz.create_batches(batch_size = cfg.batch_size, batch_type='train')

    mean_train_loss = model.train(batch_iter, optimizer)
    scheduler.step()

    print(f' || epoch {i+1} loss: {mean_train_loss:.4f}', end= '' if i != 0 else '\n')

    if i != 0:
        loss_diff = prev_train_loss - mean_train_loss
        print(f' || loss change: ↓{loss_diff:.4f}')

    prev_train_loss = mean_train_loss

    if (i + 1) % (cfg.epochs // 5) == 0:
        val_iter = tkz.create_batches(batch_size=cfg.batch_size, batch_type='val')
        mean_val_loss = model.val(val_iter)

        print(f'validation loss: {mean_val_loss:.4f}')

# Test model
test_iter = tkz.create_batches(batch_size = cfg.batch_size, batch_type='test')
mean_test_loss = model.test(test_iter)

print(f'test loss: {mean_test_loss:.4f}')

# Predictions
while 1:

    inputs = tkz.tokenize_pred(input('Enter a word: '))
    tensor = torch.tensor(inputs, dtype=torch.long, device=device).unsqueeze(0)
    encoded_ = model.pred(tensor, tkz.eow, tkz.banned_tokens, n=cfg.n, temperature=cfg.temperature, p=cfg.p, freq_penalty=cfg.freq_penalty)[0][1:]
    word = tkz.decode(encoded_)
    print(word)




