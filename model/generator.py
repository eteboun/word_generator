import torch
import torch.nn as nn
from typing import Generator
from config.config import ModelConfig

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

    @staticmethod
    def apply_sampling(logits: torch.Tensor, generated: torch.Tensor,
                       temperature: int | float, p: int | float,
                       freq_penalty: int | float) -> int:

        logits /= temperature
        logits -= freq_penalty * torch.bincount(generated, minlength=logits.shape[0])
        logits = Model.top_p_filter(logits, p=p)

        probs = torch.softmax(logits, dim=-1)
        selected = torch.multinomial(probs, 1)

        return selected.item()

    def __init__(self,
                 cfg: ModelConfig,
                 vocab_count: int,
                 pad: int) -> None:
        super().__init__()

        features = cfg.features
        hidden_state = cfg.hidden_state
        self.cfg = cfg

        self.see = nn.Parameter(torch.empty(2 * features, features))
        nn.init.kaiming_normal_(self.see)

        self.bias = nn.Parameter(torch.zeros(features))

        self.mix_in = nn.Linear(features, hidden_state)
        self.mix_out = nn.Linear(hidden_state, features)

        self.ln = nn.LayerNorm(features)

        self.emb = nn.Embedding(vocab_count, features, padding_idx=pad)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad)

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        device = inputs.device

        loss_log = []
        state = torch.zeros(inputs.shape[0], self.cfg.features, device=device)

        size = inputs.shape[-1]
        for i in range(size):

            inp_encoding = inputs[..., i]
            vecs = self.emb(inp_encoding)

            see = torch.sigmoid(torch.cat((vecs, state), dim=-1) @ self.see + self.bias)
            cont = state * see + vecs * (1 - see)

            out = self.mix_out(torch.relu(self.mix_in(self.ln(cont))))

            logits = out @ self.emb.weight.transpose(-1, -2) / self.cfg.features ** 0.5
            loss_log.append(
                self.ce_loss(logits, targets[..., i])
            )

            state = cont


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

    def predict(self, input_: torch.Tensor, state: torch.Tensor) -> tuple:

        for id_ in input_:
            vecs = self.emb(id_)

            see = torch.sigmoid(torch.cat((vecs, state), dim=-1) @ self.see + self.bias)
            cont = state * see + vecs * (1 - see)

            out = self.mix_out(torch.relu(self.mix_in(self.ln(cont))))

            logits = out @ self.emb.weight.transpose(-1, -2) / self.cfg.features ** 0.5

            state = cont

        return logits, state

    # 1-dim list
    def generate(self, encoded_input: list, eow_id: int,
                 temperature: int | float, p: int | float,
                 freq_penalty: int | float, n: int) -> list:

        if n <= 0:
            return []

        input_ = torch.tensor(encoded_input, dtype=torch.long, device=self.emb.weight.device)
        state = torch.zeros(self.cfg.features, device=self.emb.weight.device)
        generated = torch.empty(n, dtype=torch.long, device=self.emb.weight.device)
        size = 0

        logits, state = self.predict(input_, state)
        next_id = Model.apply_sampling(logits=logits, generated=generated[:size],
                                       temperature=temperature, p=p,
                                       freq_penalty=freq_penalty)

        generated[0] = next_id
        size = 1
        for _ in range(n - 1):
            if next_id == eow_id: break
            logits, state = self.predict(torch.tensor([next_id], dtype=torch.long, device=self.emb.weight.device), state)
            next_id = Model.apply_sampling(logits=logits, generated=generated[:size],
                                           temperature=temperature, p=p,
                                           freq_penalty=freq_penalty)
            generated[size] = next_id
            size += 1

        return generated[:size].tolist()
