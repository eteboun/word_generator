
import tokenizer
import generator
import torch.optim as optim
import torch

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

features = 128
hidden_state = 512
epochs = 100

lr = 0.001
temperature = 1.2
p = 0.95
freq_penalty = 1.5
n = 10

batch_size = 64

# Read data
with open('data.txt', encoding='utf-8') as f:
    data = f.read()

# Load tokenizer
tkz = tokenizer.Tokenizer()
tkz.load(data)

# Create model
model = generator.Generator(hidden_state, features, tkz).to(device)

# Run model
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)

prev_loss = 0
for i in range(epochs):
    epoch_loss = 0.0
    count = 0

    for x, y in tkz.batch(data, batch_size):
        optimizer.zero_grad()

        inputs = torch.tensor(x, dtype=torch.long, device=device)
        targets = torch.tensor(y, dtype=torch.long, device=device)

        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        count += 1
        epoch_loss += loss.item()

    scheduler.step()
    calc_loss = epoch_loss / count
    print(f'epoch {i+1} loss: {calc_loss:.4f}')

    if i != 0:
        loss_diff = prev_loss - calc_loss
        print(f'loss change: ↓{loss_diff:.4f}')

    prev_loss = calc_loss

while 1:

    inputs = tkz.tokenize_pred(input('Enter a word: '))
    tensor = torch.tensor(inputs, dtype=torch.long, device=device).unsqueeze(0)
    word = model.pred(tensor, n=n, temperature=temperature, p=p, freq_penalty=freq_penalty)
    print(word)




