import tokenizer
import generator
import config
import torch
import torch.optim as optim

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = config.Config.load('config.json')

# Read data
with open("data.txt", "r", encoding="utf-8") as f:
    data = f.read()
print("Data acquired.")

# Load tokenizer
tkz = tokenizer.Tokenizer()
tkz.load(data)
tkz.set_batch_data(data=data, train=0.7, val=0.2, test=0.1)
print("Tokenizer loaded.")

# Create model
model = generator.Generator(cfg.hidden_state, cfg.features, tkz.vocab_count, tkz.pad).to(device)

# Run model
print("Training started.")

model.train()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

prev_train_total_loss = 0.0
prev_train_data_loss = 0.0
for i in range(cfg.epochs):
    print("=======================================")

    batch_iter = tkz.create_batches(batch_size = cfg.batch_size, batch_type='train')

    train_loss_info = model.train_epoch(batch_iter, optimizer,
                                        regularization=cfg.regularization, l1_const=cfg.l1_const)
    mean_train_total_loss = train_loss_info['train_total_loss']
    mean_train_data_loss = train_loss_info['train_data_loss']

    print(f'epoch {i+1} total loss: {mean_train_total_loss:.4f}')
    print(f'epoch {i+1} data loss: {mean_train_data_loss:.4f}')


    if i != 0:
        total_loss_diff = prev_train_total_loss - mean_train_total_loss
        data_loss_diff = prev_train_data_loss - mean_train_data_loss

        print(f'total loss change: ↓{total_loss_diff:.4f}')
        print(f'data loss change: ↓{data_loss_diff:.4f}')

    prev_train_data_loss = mean_train_data_loss
    prev_train_total_loss = mean_train_total_loss

    if i % 2 == 0:
        val_iter = tkz.create_batches(batch_size=cfg.batch_size, batch_type='val')
        mean_val_loss = model.val(val_iter)

        print(f'validation loss: {mean_val_loss:.4f}')

    print("=======================================\n")

# Test model
model.eval()
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




