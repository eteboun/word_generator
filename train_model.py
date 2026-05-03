import torch
import time
import os
import shutil
import json
import torch.optim as optim
import tokenizer.tokenizer as tokenizer
import config.config as config
from model.generator import Model

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg_name = 'train1_cfg'
cfg = config.ForwardTrainConfig.load(f'./config/forward_train_cfg.json')

# Path control
if os.path.exists('./model_saves/' + cfg.paths.new_model):
    shutil.rmtree('./model_saves/' + cfg.paths.new_model)

os.makedirs(f'./model_saves/{cfg.paths.new_model}')

# Read train data
with open(f"./train_datasets/{cfg.paths.data_name}/train.txt", "r", encoding="utf-8") as f:
    train_ds = f.read()
with open(f"./train_datasets/{cfg.paths.data_name}/val.txt", "r", encoding="utf-8") as f:
    val_ds = f.read()
with open(f"./train_datasets/{cfg.paths.data_name}/test.txt", "r", encoding="utf-8") as f:
    test_ds = f.read()

# Load tokenizer
tkz = tokenizer.Tokenizer().load(f"./tokenizer/{cfg.paths.tokenizer}.json")
tkz.set_batch_data(train_ds, val_ds, test_ds)

# Load model
model_general_data = torch.load(f"./model_saves/{cfg.paths.curr_model}/info.pt", map_location=device)

model_cfg = model_general_data["model_config"]
model = Model(model_cfg, tkz.vocab_count, tkz.pad).to(device)

model_state_dict = model_general_data["model"]
model.load_state_dict(model_state_dict)

optimizer_state_dict = model_general_data["optimizer"]
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(optimizer_state_dict)

for group in optimizer.param_groups:
    group["lr"] = cfg.train.lr

# Run model
best_val = float("inf")
prev_val_loss = None
prev_data_loss = None
data_loss = model_general_data["data_loss"]
val_loss = model_general_data["val_loss"]
data_loss_diff = 0.0
val_loss_diff = 0.0
train_log = {
    'device': device,
    'config': cfg.train.get_elements()}

for i in range(cfg.train.epochs):
    epoch_log = {}

    model.train()
    batch_iter = tkz.create_batches(batch_size = cfg.train.batch_size, batch_type='train')

    start_time = time.time()
    data_loss = model.fit(batch_iter, optimizer)
    end_time = time.time()

    time_elapsed = end_time - start_time
    if i % 2 == 0:
        model.eval()
        with torch.no_grad():
            val_iter = tkz.create_batches(batch_size=cfg.train.batch_size, batch_type='val')
            val_loss = model.val(val_iter)

            val_loss_diff = val_loss - prev_val_loss if prev_val_loss is not None else 0.0
            prev_val_loss = val_loss

    data_loss_diff =  data_loss - prev_data_loss if prev_data_loss is not None else 0.0
    prev_data_loss = data_loss

    epoch_log['time'] = round(time_elapsed, 4)
    epoch_log['data_loss'] = round(data_loss, 4)
    if i != 0:
        epoch_log['data_loss_diff'] = round(data_loss_diff, 4)
        if i % 2 == 0:
            epoch_log['val_loss'] = round(val_loss, 4)
            epoch_log['val_loss_diff'] = round(val_loss_diff, 4)
    else:
        epoch_log['val_loss'] = round(val_loss, 4)

    if i % 2 == 0:
        if val_loss < best_val:
            best_val = val_loss

            # Save model
            torch.save({
                'model': model.state_dict(),
                'model_config': model.cfg.get_elements(),
                'optimizer': optimizer.state_dict(),
                'data_loss': data_loss,
                'val_loss': val_loss,
                'best_val': best_val,
            }, f'./model_saves/{cfg.paths.new_model}(info.pt')
        else:
            epoch_log['best_val'] = round(best_val, 4)
            train_log[f'epoch {i + 1}'] = epoch_log
            break

    epoch_log['best_val'] = round(best_val, 4)
    train_log[f'epoch {i + 1}'] = epoch_log

# Load last model
last = torch.load(f'./model_saves/{cfg.paths.new_model}/info.pt', map_location=device)
model.load_state_dict(last['model'])

# Test model
model.eval()

with torch.no_grad():
    test_iter = tkz.create_batches(batch_size = cfg.train.batch_size, batch_type='test')
    test_loss = model.test(test_iter)

train_log['test loss'] = round(test_loss, 4)

# Save train log
with open(f'./model_saves/{cfg.paths.new_model}/log.json', 'w') as log:
    json.dump(train_log, log, indent=4)
