import torch
import torch.nn as nn
from typing import Generator

class Model(nn.Module):

    @staticmethod
    def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        cum_sum = torch.cumsum(sorted_logits, dim=-1)

        mask = cum_sum > p

        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        logits[sorted_indices[mask]] = float('-inf')

        return logits

    def __init__(self,
                 hidden_state: int,
                 features: int,
                 vocab_count: int,
                 pad: int):
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

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor = None) -> torch.Tensor:

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

    def fit(self,
            batch_iter: Generator[tuple[list[list[int]], list[list[int]]], None, None],
            optimizer: torch.optim.Optimizer) -> float:

        epoch_data_loss = 0.0
        count = 0

        for x, y in batch_iter:
            optimizer.zero_grad()

            inputs = torch.tensor(x, dtype=torch.long, device=self.emb.weight.device)
            targets = torch.tensor(y, dtype=torch.long, device=self.emb.weight.device)

            loss = self(inputs, targets)
            epoch_data_loss += loss.item()

            loss.backward()
            optimizer.step()

            count += 1

        calc_epoch_data_loss = epoch_data_loss / count
        return calc_epoch_data_loss

    def val(self, batch_iter: Generator[tuple[list[list[int]], list[list[int]]], None, None]) -> float:
        epoch_loss = 0.0
        count = 0

        for x, y in batch_iter:

            inputs = torch.tensor(x, dtype=torch.long, device=self.emb.weight.device)
            targets = torch.tensor(y, dtype=torch.long, device=self.emb.weight.device)

            loss = self(inputs, targets)

            count += 1
            epoch_loss += loss.item()

        calc_loss = epoch_loss / count

        return calc_loss

    def test(self, batch_iter: Generator[tuple[list[list[int]], list[list[int]]], None, None]) -> float:
        batch_loss = 0.0
        count = 0

        for x, y in batch_iter:
            inputs = torch.tensor(x, dtype=torch.long, device=self.emb.weight.device)
            targets = torch.tensor(y, dtype=torch.long, device=self.emb.weight.device)

            loss = self(inputs, targets)

            count += 1
            batch_loss += loss.item()

        calc_loss = batch_loss / count

        return calc_loss