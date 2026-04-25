import torch
import torch.nn as nn

def top_p(logits, p):
    sorted_logits, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
    cum_sum = torch.cumsum(sorted_logits, dim=-1)

    mask = cum_sum > p

    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    logits[sorted_indices[mask]] = float('-inf')

    return logits

class Generator(nn.Module):
    def __init__(self, hidden_state, features, tokenizer):
        super().__init__()

        self.see = nn.Parameter(torch.empty(2 * features, features))
        nn.init.kaiming_normal_(self.see)

        self.bias = nn.Parameter(torch.zeros(features))

        self.mix_in = nn.Linear(features, hidden_state)
        self.mix_out = nn.Linear(hidden_state, features)

        self.ln = nn.LayerNorm(features)

        self.tokenizer = tokenizer
        self.emb = nn.Embedding(tokenizer.size, features, padding_idx=tokenizer.pad)

        self.features = features
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=tokenizer.pad)

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

    def pred(self, x, n, temperature=1.0, p=1.0, freq_penalty = 1.0):
        list_x = x.tolist()
        word = ''.join(self.tokenizer.decode(list_x[0][1:]))

        with torch.no_grad():
            for _ in range(n):
                curr_x = torch.tensor(list_x, dtype=torch.long, device=x.device)

                # Apply temperature
                logits = self(curr_x).squeeze(0) / temperature

                # Apply frequency penalty
                freqs = torch.bincount(curr_x[0], minlength=logits.size(-1))
                logits = logits - freq_penalty * freqs

                for token in self.tokenizer.banned_tokens:
                    logits[token] = float('-inf')

                top_p_filtered = top_p(logits, p)
                probs = torch.softmax(top_p_filtered, dim=-1)

                prediction = torch.multinomial(probs, 1)[0].item()

                if prediction == self.tokenizer.eow: break
                add = self.tokenizer.decode([prediction])

                word += add
                list_x[0].append(prediction)

        return word

