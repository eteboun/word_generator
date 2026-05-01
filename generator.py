import torch
import torch.nn as nn
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        loss = func(*args, **kwargs)
        end = time.time()

        print(f'{func.__name__} epoch took {(end - start):.3f} s')
        return loss

    return wrapper

class Generator(nn.Module):
    def __init__(self, hidden_state, features, vocab_count, pad):
        super().__init__()

        self.see = nn.Parameter(torch.empty(2 * features, features))
        nn.init.kaiming_normal_(self.see)

        self.bias = nn.Parameter(torch.zeros(features))

        self.mix_in = nn.Linear(features, hidden_state)
        self.mix_out = nn.Linear(hidden_state, features)

        self.ln = nn.LayerNorm(features)

        self.emb = nn.Embedding(vocab_count, features, padding_idx=pad)

        self.features = features
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad)

    def forward(self, inputs, targets = None):

        device = inputs.device

        loss_log = []
        state = torch.zeros(inputs.shape[0], self.features, device=device)

        size = inputs.shape[-1]
        for i in range(size):

            inp_encoding = inputs[..., i]
            vecs = self.emb(inp_encoding)

            see = torch.sigmoid(torch.cat((vecs, state), dim=-1) @ self.see + self.bias)
            cont = state * see + vecs * (1 - see)

            out = self.mix_out(torch.relu(self.mix_in(self.ln(cont))))

            logits = out @ self.emb.weight.transpose(-1, -2) / self.features ** 0.5
            if targets is not None:
                loss_log.append(
                    self.ce_loss(logits, targets[..., i])
                )

            state = cont

        if targets is None: return logits
        else:
            loss = torch.stack(loss_log).mean()
            return loss

    @timer
    def train_epoch(self, batch_iter, optimizer, regularization = "none",
                    l1_const = 0.0):
        epoch_data_loss = 0.0
        epoch_total_loss = 0.0
        count = 0

        for x, y in batch_iter:
            optimizer.zero_grad()

            inputs = torch.tensor(x, dtype=torch.long, device=self.emb.weight.device)
            targets = torch.tensor(y, dtype=torch.long, device=self.emb.weight.device)

            loss = self(inputs, targets)
            epoch_data_loss += loss.item()

            if regularization == "l1":
                l1_penalty = sum(p.abs().sum() for p in self.parameters())
                loss = loss + l1_const * l1_penalty
                epoch_total_loss += loss.item()

            loss.backward()
            optimizer.step()

            count += 1

        calc_epoch_data_loss = epoch_data_loss / count
        calc_epoch_total_loss = epoch_total_loss / count

        return {
            'train_data_loss': calc_epoch_data_loss,
            'train_total_loss': calc_epoch_total_loss,
        }

    def val(self, batch_iter):
        epoch_loss = 0.0
        count = 0

        self.eval()
        with torch.no_grad():
            for x, y in batch_iter:

                inputs = torch.tensor(x, dtype=torch.long, device=self.emb.weight.device)
                targets = torch.tensor(y, dtype=torch.long, device=self.emb.weight.device)

                loss = self(inputs, targets)

                count += 1
                epoch_loss += loss.item()

            calc_loss = epoch_loss / count

        self.eval()
        return calc_loss

    def test(self, batch_iter):
        batch_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, y in batch_iter:
                inputs = torch.tensor(x, dtype=torch.long, device=self.emb.weight.device)
                targets = torch.tensor(y, dtype=torch.long, device=self.emb.weight.device)

                loss = self(inputs, targets)

                count += 1
                batch_loss += loss.item()

            calc_loss = batch_loss / count

        return calc_loss

    @staticmethod
    def top_p_filter(logits, p):
        sorted_logits, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        cum_sum = torch.cumsum(sorted_logits, dim=-1)

        mask = cum_sum > p

        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        logits[sorted_indices[mask]] = float('-inf')

        return logits

    def pred(self, x, stop_token, banned_tokens, n, temperature=1.0, p=1.0, freq_penalty = 1.0):
        list_x = x.tolist()
        with torch.no_grad():
            for _ in range(n):
                curr_x = torch.tensor(list_x, dtype=torch.long, device=x.device)

                # Apply temperature
                logits = self(curr_x).squeeze(0) / temperature

                # Apply frequency penalty
                freqs = torch.bincount(curr_x[0], minlength=logits.size(-1))
                logits = logits - freq_penalty * freqs

                # Mask
                for token in banned_tokens:
                    logits[token] = float('-inf')

                # Apply top-p filtering
                top_p_filtered = Generator.top_p_filter(logits, p)
                probs = torch.softmax(top_p_filtered, dim=-1)

                # Predict
                prediction = torch.multinomial(probs, 1)[0].item()

                if prediction == stop_token: break
                list_x[0].append(prediction)

        return list_x

